import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.adapters.base import BaseAdapter
from src.core.logger import get_logger

from .core.monitor import BenchmarkMonitor
from .core.metrics import MetricsCalculator
from .core.judge_util import llm_grader

from concurrent.futures import TimeoutError
from tqdm import tqdm

class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db, llm):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()

        self.output_dir = self.config['paths']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")

        self.metrics_summary = {
            "insertion": {},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0},
            "judge": {"time": 0, "input_tokens": 0, "output_tokens": 0},
        }

    # =========================================================
    # ===================== GENERATION ========================
    # =========================================================

    def run_generation(self):
        self.logger.info(">>> Stage: Ingestion & Generation")

        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        doc_info = self.adapter.data_prepare(doc_dir)

        skip_ingestion = self.config['execution'].get('skip_ingestion', False)

        if skip_ingestion:
            self.logger.info("Skipping Ingestion.")
        else:
            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            ingest_stats = self.db.ingest(doc_info, max_workers=ingest_workers, monitor=self.monitor)

            # 保留完整 ingest_stats（含 breakdown）
            self.metrics_summary["insertion"] = ingest_stats

            payload = {
                "Insertion Efficiency (Total Dataset)": ingest_stats
            }
            self._update_report(payload)

        samples = self.adapter.load_and_transform()
        tasks = self._prepare_tasks(samples)

        results_map = {}
        max_workers = self.config['execution']['max_workers']

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_generation_task, task): task
                for task in tasks
            }

            pbar = tqdm(total=len(tasks), desc="Generating Answers", unit="task")
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Generation failed for task {task['id']}: {e}")
                    self.monitor.worker_end(success=False)
                pbar.update(1)
            pbar.close()

        sorted_results = [results_map[i] for i in sorted(results_map.keys())]

        save_data = {
            "summary": {
                "dataset": self.config.get('dataset_name'),
                "total_queries": len(sorted_results),
                "insertion_efficiency": self.metrics_summary["insertion"],
            },
            "results": sorted_results
        }

        # 写平均效率
        total = len(sorted_results)
        if total > 0:
            self._update_report({
                "Query Efficiency (Average Per Query)": {
                    "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in sorted_results) / total,
                    "Average Input Tokens (cl100k_base)": sum(r['token_usage']['total_input_tokens'] for r in sorted_results) / total,
                    "Average Output Tokens (cl100k_base)": sum(r['token_usage']['llm_output_tokens'] for r in sorted_results) / total,
                    "Average HippoRAG2 Internal Input Tokens (cl100k_base)": sum(r['token_usage'].get('hippo_internal_input_tokens', 0) for r in sorted_results) / total,
                    "Average HippoRAG2 Internal Output Tokens (cl100k_base)": sum(r['token_usage'].get('hippo_internal_output_tokens', 0) for r in sorted_results) / total,
                    "Average Generation Prompt Tokens (cl100k_base)": sum(r['token_usage']['generation_usage'].get('cl100k_prompt_tokens', 0) for r in sorted_results) / total,
                    "Average Generation Completion Tokens (cl100k_base)": sum(r['token_usage']['generation_usage'].get('cl100k_completion_tokens', 0) for r in sorted_results) / total,
                }
            })

        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    # =========================================================
    # ==================== PER QUERY TASK =====================
    # =========================================================

    def _process_generation_task(self, task):
        self.monitor.worker_start()
        qa = task['qa']

        # ---------- Retrieval ----------
        t0 = time.time()
        search_res = self.db.retrieve(
            query=qa.question,
            topk=self.config['execution']['retrieval_topk']
        )
        latency = time.time() - t0

        hippo_delta = {}
        if hasattr(self.db, "get_last_retrieve_delta"):
            hippo_delta = self.db.get_last_retrieve_delta()

        hippo_internal_in = int(
            hippo_delta.get("embed_input_tokens", 0)
            + hippo_delta.get("llm_prompt_tokens", 0)
        )
        hippo_internal_out = int(
            hippo_delta.get("llm_completion_tokens", 0)
        )

        retrieved_texts = []
        retrieved_uris = []
        context_blocks = []

        for r in search_res.resources:
            retrieved_uris.append(r.uri)
            content = self.db.read_resource(r.uri)
            retrieved_texts.append(content)
            context_blocks.append(content[:2000])

        recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

        # ---------- Prompt ----------
        full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)

        # ---------- Generation ----------
        ans_raw = self.llm.generate(full_prompt)
        generation_usage = getattr(self.llm, "last_usage", {})

        ans = self.adapter.post_process_answer(qa, ans_raw, meta)

        # ---------- Token stats ----------
        in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)
        out_tokens = self.db.count_tokens(ans)

        self.monitor.worker_end(tokens=in_tokens + out_tokens)

        return {
            "_global_index": task['id'],
            "sample_id": task['sample_id'],
            "question": qa.question,
            "gold_answers": qa.gold_answers,
            "category": str(qa.category),
            "evidence": qa.evidence,
            "retrieval": {
                "latency_sec": latency,
                "uris": retrieved_uris
            },
            "llm": {"final_answer": ans},
            "metrics": {"Recall": recall},
            "token_usage": {
                # ruc 口径
                "total_input_tokens": int(in_tokens),
                "llm_output_tokens": int(out_tokens),

                # generation 真实 usage
                "generation_usage": generation_usage,

                # HippoRAG2 内部
                "hippo_internal_input_tokens": hippo_internal_in,
                "hippo_internal_output_tokens": hippo_internal_out,
                "hippo_internal_detail": hippo_delta,
            }
        }

    # =========================================================
    # ====================== EVALUATION =======================
    # =========================================================
    def run_evaluation(self):
        from concurrent.futures import TimeoutError  # ✅ 记得导入

        if not os.path.exists(self.generated_file):
            self.logger.error(f"Generated answers file not found: {self.generated_file}")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            items = json.load(f)["results"]

        # ✅ 让 max_queries 对 eval 生效：必须在 submit 之前截断
        max_q = self.config.get("execution", {}).get("max_queries")
        if max_q is not None:
            try:
                max_q = int(max_q)
                items = items[:max_q]
            except Exception:
                self.logger.warning(f"Invalid max_queries={max_q}, ignore slicing.")

        self.logger.info(f"Stage: Evaluation (items={len(items)})")

        timeout_s = self.config.get("execution", {}).get("eval_task_timeout", 120)

        results = {}
        partial_file = self.eval_file.replace(".json", ".partial.json")

        with ThreadPoolExecutor(max_workers=self.config["execution"]["max_workers"]) as executor:
            future_map = {
                executor.submit(self._process_evaluation_task, item): item
                for item in items
            }

            pbar = tqdm(total=len(future_map), desc="Evaluating", unit="item")

            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    res = future.result(timeout=timeout_s)
                except TimeoutError:
                    self.logger.error(
                        f"[EVAL TIMEOUT] sample_id={item.get('sample_id')} "
                        f"q={str(item.get('question',''))[:80]}"
                    )
                    # 兜底：判错 + 写回
                    item.setdefault("metrics", {})
                    item["metrics"].update({
                        "F1": item.get("metrics", {}).get("F1", 0.0),
                        "Accuracy": 0.0,
                        "judge_usage": {"error": "timeout"}
                    })
                    res = item
                except Exception as e:
                    self.logger.error(
                        f"[EVAL ERROR] sample_id={item.get('sample_id')} err={e}"
                    )
                    item.setdefault("metrics", {})
                    item["metrics"].update({
                        "F1": item.get("metrics", {}).get("F1", 0.0),
                        "Accuracy": 0.0,
                        "judge_usage": {"error": str(e)}
                    })
                    res = item

                results[res["_global_index"]] = res
                pbar.update(1)

                # ✅ 边跑边写 partial，方便你立刻看到初步输出
                try:
                    eval_records_partial = [results[k] for k in sorted(results.keys())]
                    with open(partial_file, "w", encoding="utf-8") as pf:
                        json.dump({"results": eval_records_partial}, pf, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.warning(f"Failed to write partial eval file: {e}")

            pbar.close()

        eval_records = [results[k] for k in sorted(results.keys())]

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Eval results saved -> {self.eval_file}")
        self.logger.info(f"Partial results saved -> {partial_file}")

        self._update_report({
            "Judge Efficiency (Total)": self.metrics_summary["judge"]
        })

    def _process_evaluation_task(self, item):
        ans = item["llm"]["final_answer"]
        golds = item["gold_answers"]

        f1 = max((MetricsCalculator.calculate_f1(ans, g) for g in golds), default=0.0)

        dataset_name = self.config.get('dataset_name')

        t0 = time.time()

        ok = llm_grader(
            self.llm,
            self.config['llm']['model'],
            item["question"],
            "\n".join(golds),
            ans,
            dataset_name=dataset_name
        )

        dt = time.time() - t0

        # judge 使用 LLMClientWrapper.last_usage 精确记录
        judge_usage = getattr(self.llm, "last_usage", {})

        self.metrics_summary["judge"]["time"] += dt
        self.metrics_summary["judge"]["input_tokens"] += judge_usage.get("cl100k_prompt_tokens", 0)
        self.metrics_summary["judge"]["output_tokens"] += judge_usage.get("cl100k_completion_tokens", 0)

        item["metrics"].update({
            "F1": f1,
            "Accuracy": 1.0 if ok else 0.0,
            "judge_usage": judge_usage
        })

        return item

    # =========================================================

    def _prepare_tasks(self, samples):
        tasks = []
        idx = 0
        max_queries = self.config['execution'].get('max_queries')

        for sample in samples:
            for qa in sample.qa_pairs:
                if max_queries is not None and idx >= max_queries:
                    break
                tasks.append({
                    "id": idx,
                    "sample_id": sample.sample_id,
                    "qa": qa
                })
                idx += 1
        return tasks

    def _update_report(self, data):
        report = {}
        if os.path.exists(self.report_file):
            with open(self.report_file, "r", encoding="utf-8") as f:
                try:
                    report = json.load(f)
                except:
                    report = {}

        report.update(data)

        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        self.logger.info(f"Report updated -> {self.report_file}")
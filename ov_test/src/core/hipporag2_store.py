import os
import time
import json
import shutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable

import tiktoken

from src.adapters.base import StandardDoc


@dataclass
class _Resource:
    uri: str
    is_leaf: bool = True
    abstract: str = ""
    overview: str = ""


@dataclass
class _SearchResult:
    resources: List[_Resource]


class HippoRAG2StoreWrapper:
    """
    让 ruc pipeline 把 HippoRAG2 当成“vector store”用，并尽量完整统计 token/time：
    - ingest(): 读 adapter 生成的 md 文档 -> hipporag.index(docs)
      * 统计：doc 输入 token + HippoRAG2 内部 LLM token（优先读 usage，兜底估算）+ embedding 输入 token（估算）
    - retrieve(): hipporag.retrieve([query])
      * 统计：query embedding 输入 token（估算）+ 内部 rerank/LLM token（如有）
    - read_resource(): 根据 uri 返回对应文本块
    - count_tokens(): cl100k_base
    - clear(): 真正删除 store_path（对齐 deletion 语义）
    """

    def __init__(self, store_path: str, hippo_cfg: Optional[dict] = None):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)

        self.enc = tiktoken.get_encoding("cl100k_base")
        self._last_retrieved: Dict[str, str] = {}

        hippo_cfg = hippo_cfg or {}
        self.hippo_cfg = hippo_cfg

        # --- 统计累加器（全程累计，可在 ingest/retrieve 后读取）---
        self._counters: Dict[str, int] = {
            # LLM
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_calls": 0,

            # Embedding（估算：输入文本 token）
            "embed_input_tokens": 0,
            "embed_calls": 0,
        }

        # --- HippoRAG2 初始化 ---
        from src.hipporag.HippoRAG import HippoRAG

        self.hipporag = HippoRAG(
            save_dir=hippo_cfg.get("save_dir", store_path),
            llm_model_name=hippo_cfg.get("llm_model_name", "gpt-4o-mini"),
            llm_base_url=hippo_cfg.get("llm_base_url"),
            embedding_model_name=hippo_cfg.get("embedding_model_name", "facebook/contriever"),
            embedding_base_url=hippo_cfg.get("embedding_base_url"),
        )

        # retrieve topk 默认值（如果上层给 topk，就以 topk 为准）
        self.num_to_retrieve_default = int(hippo_cfg.get("num_to_retrieve", 5))

        # --- 尝试对 HippoRAG2 内部 client 做 monkeypatch（防止 token 漏计）---
        self._install_usage_hooks()

    # -------------------------
    # Token helpers
    # -------------------------
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.enc.encode(str(text)))

    def _count_message_tokens_cl100k(self, messages: List[Dict[str, Any]]) -> int:
        """
        OpenAI chat messages 的粗略 cl100k 估算。
        不追求与官方完全一致，但保证不漏（偏保守）。
        """
        if not messages:
            return 0
        total = 0
        for m in messages:
            # role/name 的 token 影响很小，这里把 content + role 字符串一起计入，偏保守
            total += self.count_tokens(m.get("role", ""))
            total += self.count_tokens(m.get("content", ""))
            if "name" in m:
                total += self.count_tokens(m.get("name", ""))
        return total

    def _bump(self, k: str, v: int) -> None:
        if v is None:
            return
        self._counters[k] = int(self._counters.get(k, 0)) + int(v)

    def _snapshot_and_reset_internal_usage_if_any(self) -> Dict[str, int]:
        """
        如果 HippoRAG2 的 llm_model 提供 get_and_reset_usage()，优先用它。
        期望返回类似：{"prompt_tokens":..., "completion_tokens":..., "calls":..., ...}
        """
        llm_model = getattr(self.hipporag, "llm_model", None)
        if llm_model is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

        if hasattr(llm_model, "get_and_reset_usage") and callable(getattr(llm_model, "get_and_reset_usage")):
            try:
                u = llm_model.get_and_reset_usage()
                pt = int(u.get("prompt_tokens", 0))
                ct = int(u.get("completion_tokens", 0))
                calls = int(u.get("calls", 0))
                return {"prompt_tokens": pt, "completion_tokens": ct, "calls": calls}
            except Exception:
                return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

        return {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    # -------------------------
    # Monkeypatch hooks
    # -------------------------
    def _install_usage_hooks(self) -> None:
        """
        目的：尽量捕捉 HippoRAG2 内部所有调用的 token。
        - LLM：尝试 hook openai_client.chat.completions.create
        - Embedding：尝试 hook embedding 模型的 encode / embed / get_embeddings 等
        """
        # 1) LLM hook（OpenAI-compatible client）
        llm_model = getattr(self.hipporag, "llm_model", None)
        openai_client = getattr(llm_model, "openai_client", None) if llm_model else None

        try:
            if openai_client and hasattr(openai_client, "chat") and hasattr(openai_client.chat, "completions"):
                completions = openai_client.chat.completions
                if hasattr(completions, "create") and callable(completions.create):
                    orig_create = completions.create

                    def wrapped_create(*args, **kwargs):
                        # 估算 prompt tokens
                        messages = kwargs.get("messages", None)
                        if messages is None and len(args) >= 1:
                            # 不同 SDK 可能把 messages 放 args，这里仅兜底
                            messages = None
                        prompt_tokens_est = 0
                        if isinstance(messages, list):
                            prompt_tokens_est = self._count_message_tokens_cl100k(messages)

                        # 调用原方法
                        resp = orig_create(*args, **kwargs)

                        # 优先用 resp.usage（如果 provider 返回）
                        usage = getattr(resp, "usage", None)
                        if usage is not None:
                            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                            ct = int(getattr(usage, "completion_tokens", 0) or 0)
                        else:
                            # 兜底：用返回 content 估算 completion_tokens
                            pt = prompt_tokens_est
                            ct = 0
                            try:
                                content = resp.choices[0].message.content
                                ct = self.count_tokens(content)
                            except Exception:
                                ct = 0

                        self._bump("llm_prompt_tokens", pt)
                        self._bump("llm_completion_tokens", ct)
                        self._bump("llm_calls", 1)

                        return resp

                    completions.create = wrapped_create
        except Exception:
            # hook 失败也不致命，后续还能靠 get_and_reset_usage 或外部估算
            pass

        # 2) Embedding hook（尽量泛化）
        emb_model = getattr(self.hipporag, "embedding_model", None)
        if emb_model is None:
            return

        # 常见命名：encode / embed_documents / get_embeddings / __call__
        candidate_methods = ["encode", "embed", "get_embeddings", "embed_documents", "embedding", "__call__"]
        for mname in candidate_methods:
            if hasattr(emb_model, mname) and callable(getattr(emb_model, mname)):
                orig = getattr(emb_model, mname)

                def make_wrapper(orig_fn: Callable):
                    def wrapped(texts, *args, **kwargs):
                        # texts 可能是 str / list[str]
                        input_tokens = 0
                        if isinstance(texts, str):
                            input_tokens = self.count_tokens(texts)
                            self._bump("embed_calls", 1)
                        elif isinstance(texts, list):
                            input_tokens = sum(self.count_tokens(t) for t in texts)
                            self._bump("embed_calls", 1)
                        # 累计估算 embedding 输入 token
                        self._bump("embed_input_tokens", input_tokens)
                        return orig_fn(texts, *args, **kwargs)
                    return wrapped

                try:
                    setattr(emb_model, mname, make_wrapper(orig))
                    break
                except Exception:
                    continue

    # -------------------------
    # Public APIs for pipeline
    # -------------------------
    def get_counters_snapshot(self) -> Dict[str, int]:
        """给 pipeline 用：取当前累计 counters（不 reset）"""
        return dict(self._counters)

    def reset_counters(self) -> None:
        for k in list(self._counters.keys()):
            self._counters[k] = 0

    def ingest(self, samples: List[StandardDoc], max_workers=10, monitor=None) -> dict:
        """
        ruc insertion efficiency 读取返回 dict 的 time/input_tokens/output_tokens。
        我们扩展增加 breakdown 字段（pipeline 可以写入 report）。
        """
        # ingest 前先清空计数器（便于 insertion 的统计是“本次 ingest 的”）
        self.reset_counters()

        t0 = time.time()

        docs: List[str] = []
        doc_input_tokens = 0

        for s in samples:
            with open(s.doc_path, "r", encoding="utf-8") as f:
                txt = f.read()
            docs.append(txt)
            doc_input_tokens += self.count_tokens(txt)

        # 主调用：HippoRAG2 indexing
        self.hipporag.index(docs=docs)

        # 优先读取 HippoRAG2 内部 usage（若有）
        usage = self._snapshot_and_reset_internal_usage_if_any()
        if usage["calls"] > 0:
            # 用内部 usage 覆盖/补充（更准）
            self._bump("llm_prompt_tokens", usage["prompt_tokens"])
            self._bump("llm_completion_tokens", usage["completion_tokens"])
            self._bump("llm_calls", usage["calls"])

        duration = time.time() - t0

        counters = self.get_counters_snapshot()

        # 这里的 input_tokens 需要“对齐口径”：我们把 doc 文本 + embedding 输入 + llm prompt 都算入 input
        # output_tokens：按 ruc 语义通常指 llm completion
        input_tokens_total = int(doc_input_tokens + counters["embed_input_tokens"] + counters["llm_prompt_tokens"])
        output_tokens_total = int(counters["llm_completion_tokens"])

        return {
            "time": duration,
            "input_tokens": input_tokens_total,
            "output_tokens": output_tokens_total,
            "breakdown": {
                "doc_input_tokens": int(doc_input_tokens),
                "embed_input_tokens": int(counters["embed_input_tokens"]),
                "llm_prompt_tokens": int(counters["llm_prompt_tokens"]),
                "llm_completion_tokens": int(counters["llm_completion_tokens"]),
                "llm_calls": int(counters["llm_calls"]),
                "embed_calls": int(counters["embed_calls"]),
            },
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "hippo://resources"):
        """
        返回结构要有 .resources，resource 有 uri。
        同时在 wrapper 内部累计 retrieval 内部 token（query embed + rerank llm 等）。
        """
        self._last_retrieved.clear()

        # 对 retrieval 阶段，我们不清空全局 counters（让 pipeline 能累计全程），
        # 但会记录“本次 retrieve 的增量”，pipeline 可选取增量写入 per-query。
        before = self.get_counters_snapshot()

        # 注意：不要用 min(topk, default)，否则会悄悄截断，和 ruc/topk 对不上
        k = int(topk) if topk is not None else int(self.num_to_retrieve_default)

        res = self.hipporag.retrieve(queries=[query], num_to_retrieve=k)

        # 读取内部 usage（若有）
        usage = self._snapshot_and_reset_internal_usage_if_any()
        if usage["calls"] > 0:
            self._bump("llm_prompt_tokens", usage["prompt_tokens"])
            self._bump("llm_completion_tokens", usage["completion_tokens"])
            self._bump("llm_calls", usage["calls"])

        # 解析返回
        texts: List[str] = []
        if isinstance(res, list) and len(res) > 0:
            first = res[0]
            if isinstance(first, list):
                texts = [str(x) for x in first][:k]
            else:
                texts = [str(first)][:k]

        resources: List[_Resource] = []
        for i, t in enumerate(texts):
            uri = f"{target_uri}/{i}"
            self._last_retrieved[uri] = t
            resources.append(_Resource(uri=uri, is_leaf=True))

        after = self.get_counters_snapshot()
        # 计算增量（本次 retrieve 新增的内部 token）
        delta = {key: int(after.get(key, 0) - before.get(key, 0)) for key in after.keys()}
        # 把 delta 放在对象上，pipeline 可读取
        self._last_retrieve_delta = delta

        return _SearchResult(resources=resources)

    def get_last_retrieve_delta(self) -> Dict[str, int]:
        """返回最近一次 retrieve 产生的内部 token 增量（用于 per-query 统计）"""
        return getattr(self, "_last_retrieve_delta", {"llm_prompt_tokens": 0, "llm_completion_tokens": 0, "embed_input_tokens": 0, "llm_calls": 0, "embed_calls": 0})

    def read_resource(self, uri: str) -> str:
        return self._last_retrieved.get(uri, "")

    def clear(self):
        """
        对齐 deletion：真正清掉 store_path（以及 HippoRAG2 save_dir 对应目录）。
        注意：Windows 上删除打开文件会失败，必要时先关闭程序再删。
        """
        self._last_retrieved.clear()

        # 尽量删除 store_path 目录内容
        try:
            if os.path.exists(self.store_path):
                shutil.rmtree(self.store_path)
        except Exception:
            # 容错：退化为清空目录内文件
            try:
                for root, dirs, files in os.walk(self.store_path):
                    for fn in files:
                        try:
                            os.remove(os.path.join(root, fn))
                        except Exception:
                            pass
            except Exception:
                pass
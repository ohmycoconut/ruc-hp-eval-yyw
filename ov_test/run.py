import os
import sys
import yaml
import importlib
from argparse import ArgumentParser
from src.core.logger import setup_logging

# ==========================================
# 1. 路径初始化
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
PROJECT_ROOT = WORKSPACE_ROOT

# ==========================================
# 2. 辅助函数
# ==========================================

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def resolve_path(path_str, base_path):
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_path, path_str))


# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = ArgumentParser(description="Run HippoRAG2 Benchmark (No OpenViking)")
    default_config_path = os.path.join(SCRIPT_DIR, "config/config.yaml")

    parser.add_argument("--config", default=default_config_path)
    parser.add_argument("--step", choices=["all", "gen", "eval"], default="all")

    args = parser.parse_args()

    # ====== 加载 config ======
    config_path = os.path.abspath(args.config)
    print(f"[Init] Loading configuration from: {config_path}")

    config = load_config(config_path)

    dataset_name = config.get("dataset_name", "UnknownDataset")

    # ====== 修正路径 ======
    path_keys = ['raw_data', 'output_dir', 'vector_store', 'log_file', 'doc_output_dir']
    for key in path_keys:
        if key in config.get('paths', {}):
            original = config['paths'][key]
            rendered = original.format(dataset_name=dataset_name)
            resolved = resolve_path(rendered, PROJECT_ROOT)
            config['paths'][key] = resolved

    # ====== 初始化 logger ======
    logger = setup_logging(config['paths']['log_file'])
    logger.info(">>> HippoRAG2 Benchmark Started")

    # ==========================================
    # 1️⃣ Adapter
    # ==========================================
    adapter_cfg = config.get('adapter', {})
    module_path = adapter_cfg.get('module', 'src.adapters.locomo_adapter')
    class_name = adapter_cfg.get('class_name', 'LocomoAdapter')

    mod = importlib.import_module(module_path)
    AdapterClass = getattr(mod, class_name)
    adapter = AdapterClass(raw_file_path=config['paths']['raw_data'])

    # ==========================================
    # 2️⃣ 加载 HippoRAG2 仓库路径
    # ==========================================
    hippo_repo = config.get("hipporag2", {}).get(
        "repo_root",
        r"C:\Users\29352\Desktop\programming\Windows\locomo\Hippo\HippoRAG"
    )

    if hippo_repo not in sys.path:
        sys.path.append(hippo_repo)

    # ==========================================
    # 3️⃣ 初始化 HippoRAG2 Backend
    # ==========================================
    from src.core.hipporag2_store import HippoRAG2StoreWrapper

    vector_store = HippoRAG2StoreWrapper(
        store_path=config['paths']['vector_store'],
        hippo_cfg=config.get("hipporag2", {})
    )

    # ==========================================
    # 4️⃣ LLM Client（ruc 用于最终生成）
    # ==========================================
    from src.core.llm_client import LLMClientWrapper

    api_key = os.environ.get(
        config['llm'].get('api_key_env_var', ''),
        config['llm'].get('api_key')
    )

    llm_client = LLMClientWrapper(config=config['llm'], api_key=api_key)

    # ==========================================
    # 5️⃣ Pipeline
    # ==========================================
    from src.pipeline import BenchmarkPipeline

    pipeline = BenchmarkPipeline(
        config=config,
        adapter=adapter,
        vector_db=vector_store,
        llm=llm_client
    )

    # ==========================================
    # 6️⃣ 执行
    # ==========================================
    if args.step in ["all", "gen"]:
        logger.info("Stage: Generation")
        pipeline.run_generation()

    if args.step in ["all", "eval"]:
        logger.info("Stage: Evaluation")
        pipeline.run_evaluation()

    logger.info("Benchmark finished successfully.")


if __name__ == "__main__":
    main()
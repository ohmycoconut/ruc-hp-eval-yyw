import time
from typing import Dict, Any, Optional

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=api_key,
            base_url=config['base_url'],
            timeout=config.get("timeout", 60),          # 关键：请求超时
            max_retries=config.get("max_retries", 2),   # 关键：底层重试
        )
        
        self.retry_count = 3

        # --- token accounting (cl100k_base) ---
        self.enc = tiktoken.get_encoding("cl100k_base")

        # last call usage + total usage accumulator
        self.last_usage: Dict[str, Any] = {}
        self.usage_total: Dict[str, int] = {
            "calls": 0,
            "cl100k_prompt_tokens": 0,
            "cl100k_completion_tokens": 0,
            "cl100k_total_tokens": 0,
        }

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.enc.encode(str(text)))

    def generate(self, prompt: str) -> str:
        """调用 LLM 生成回答，包含简单的指数退避重试。返回 content，不改变原接口。"""
        last_err = None
        self.last_usage = {}

        # cl100k prompt tokens（统一口径：你要求的）
        cl_prompt_tokens = self._count_tokens(prompt)

        for attempt in range(self.retry_count):
            try:
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                content = resp.content

                # cl100k completion tokens（统一口径）
                cl_completion_tokens = self._count_tokens(content)

                # provider usage（如果 LangChain 带回来了，就记录；不保证一定有）
                # 常见位置：resp.usage_metadata 或 resp.response_metadata
                provider_usage = None
                try:
                    provider_usage = getattr(resp, "usage_metadata", None)
                    if provider_usage is None:
                        provider_usage = resp.response_metadata.get("token_usage") if hasattr(resp, "response_metadata") else None
                except Exception:
                    provider_usage = None

                self.last_usage = {
                    "calls": 1,
                    "cl100k_prompt_tokens": int(cl_prompt_tokens),
                    "cl100k_completion_tokens": int(cl_completion_tokens),
                    "cl100k_total_tokens": int(cl_prompt_tokens + cl_completion_tokens),
                    "provider_usage": provider_usage,
                    "model": getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None),
                }

                # accumulate
                self.usage_total["calls"] += 1
                self.usage_total["cl100k_prompt_tokens"] += int(cl_prompt_tokens)
                self.usage_total["cl100k_completion_tokens"] += int(cl_completion_tokens)
                self.usage_total["cl100k_total_tokens"] += int(cl_prompt_tokens + cl_completion_tokens)

                return content

            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

        # 失败也记录一次（方便排查）
        self.last_usage = {
            "calls": 0,
            "error": str(last_err),
            "cl100k_prompt_tokens": int(cl_prompt_tokens),
            "cl100k_completion_tokens": 0,
            "cl100k_total_tokens": int(cl_prompt_tokens),
        }
        return f"ERROR: {str(last_err)}"
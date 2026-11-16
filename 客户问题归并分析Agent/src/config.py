"""全局配置：所有可变参数都从环境变量读取，避免把 key 写进代码。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
TRACE_DIR = DATA_DIR / "traces"
for _d in (DATA_DIR, TRACE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- LLM ----
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def resolve_llm() -> dict:
    """优先 DeepSeek，其次 OpenAI，都没有则返回 enabled=False（离线规则模式）。"""
    if DEEPSEEK_API_KEY:
        return {
            "enabled": True,
            "provider": "deepseek",
            "api_key": DEEPSEEK_API_KEY,
            "base_url": DEEPSEEK_BASE_URL,
            "model": DEEPSEEK_MODEL,
        }
    if OPENAI_API_KEY:
        return {
            "enabled": True,
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "base_url": OPENAI_BASE_URL,
            "model": OPENAI_MODEL,
        }
    return {"enabled": False, "provider": "offline", "api_key": "", "base_url": "", "model": "rule-based"}


# ---- 记忆 / 向量库 ----
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "agent_memory.db")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "qa_memory")

# ---- Agent 参数 ----
CLUSTER_THRESHOLD = float(os.getenv("CLUSTER_THRESHOLD", "0.55"))
MAX_REFINE_ROUNDS = int(os.getenv("MAX_REFINE_ROUNDS", "1"))

# ---- 可观测性 ----
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

APP_TITLE = "客户问题归并分析 Agent"
APP_SUBTITLE = "上传群聊记录 → 自动抽取/合并高频问题 → 标注优先级 → 表格化填写回答"

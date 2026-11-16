"""可观测性：Langfuse Trace（配置了 key 时启用）+ 本地 JSONL Trace（始终可用）。"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Dict, List

from . import config

_observe = None
if config.LANGFUSE_ENABLED:
    try:  # pragma: no cover - 取决于环境是否配置 key
        from langfuse import observe as _observe  # type: ignore
    except Exception:  # noqa: BLE001
        _observe = None


def observe_node(name: str) -> Callable:
    """给 LangGraph 节点加 Langfuse span；未配置时原样返回，保证可离线运行。"""

    def decorator(fn: Callable) -> Callable:
        if _observe is None:
            return fn
        try:
            return _observe(name=name)(fn)
        except Exception:  # noqa: BLE001
            return fn

    return decorator


def langfuse_ready() -> bool:
    return _observe is not None


def write_local_trace(session_id: str, trace: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> str:
    """把决策轨迹落到 data/traces/{session_id}.jsonl，便于离线复盘与报告举证。"""
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TRACE_DIR / f"{session_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for item in trace:
            fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **(extra or {}), **item}, ensure_ascii=False) + "\n")
    return str(path)


def flush() -> None:
    if _observe is None:
        return
    try:  # pragma: no cover
        from langfuse import get_client

        get_client().flush()
    except Exception:  # noqa: BLE001
        pass

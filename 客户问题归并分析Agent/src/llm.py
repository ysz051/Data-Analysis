"""LLM 客户端：DeepSeek / OpenAI（OpenAI 兼容协议）+ 无 Key 时的确定性降级。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import config

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(raw: str) -> Optional[Any]:
    """容错解析：去掉 ``` 包裹，截取第一个完整 JSON 对象/数组。"""
    if not raw:
        return None
    text = raw.strip()
    fenced = _JSON_BLOCK.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    for end in range(len(text), start, -1):
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


class LLMClient:
    """统一入口：enabled=False 时所有方法返回 None，由调用方走规则分支。"""

    def __init__(self) -> None:
        info = config.resolve_llm()
        self.enabled: bool = info["enabled"]
        self.provider: str = info["provider"]
        self.model: str = info["model"]
        self._info = info
        self._chat = None
        self.last_error: str = ""

    # ---- 基础调用 ----
    def _client(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(
                model=self._info["model"],
                api_key=self._info["api_key"],
                base_url=self._info["base_url"],
                temperature=0,
                timeout=60,
                max_retries=1,
            )
        return self._chat

    def chat(self, system: str, user: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            msg = self._client().invoke([("system", system), ("human", user)])
            content = msg.content
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content or "")
        except Exception as exc:  # noqa: BLE001 - 网络/额度异常一律降级
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def json_chat(self, system: str, user: str) -> Optional[Any]:
        raw = self.chat(system, user)
        if raw is None:
            return None
        return extract_json(raw)

    # ---- 业务语义封装 ----
    def extract_questions(self, turns: List[Dict], business_context: str = "") -> Optional[List[Dict]]:
        from .prompts import EXTRACT_SYSTEM, build_extract_user

        payload = self.json_chat(EXTRACT_SYSTEM, build_extract_user(turns, business_context))
        if isinstance(payload, dict):
            payload = payload.get("questions")
        if not isinstance(payload, list):
            return None
        out: List[Dict] = []
        for item in payload:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"text": item})
        return out or None

    def name_clusters(self, clusters: List[Dict], business_context: str = "") -> Optional[List[Dict]]:
        from .prompts import MERGE_SYSTEM, build_merge_user

        payload = self.json_chat(MERGE_SYSTEM, build_merge_user(clusters, business_context))
        if isinstance(payload, dict):
            payload = payload.get("clusters")
        return payload if isinstance(payload, list) else None

    def draft_answer(self, question: str, variants: List[str], history: List[str], business_context: str = "") -> Optional[str]:
        from .prompts import ANSWER_SYSTEM, build_answer_user

        return self.chat(ANSWER_SYSTEM, build_answer_user(question, variants, history, business_context))

    @property
    def mode_label(self) -> str:
        return f"{self.provider}:{self.model}" if self.enabled else "offline:rule-based"


_shared: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _shared
    if _shared is None:
        _shared = LLMClient()
    return _shared

"""LangGraph 状态图：感知 → 决策 → 行动（工具）→ 观察 → 人类确认 → 记忆入库。

节点职责单一，路由由 plan 节点根据当前 State 自主决定下一次 Action，
体现 Agent 的「自主决策 + 工具调用」而非固定流水线。
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import config, memory, observability
from .llm import get_llm
from .schemas import AgentResult, Cluster
from .tools import (
    new_session_id,
    run_tool,
)


class AgentState(TypedDict, total=False):
    session_id: str
    file_name: str
    file_bytes: Optional[bytes]
    pasted_text: str
    business_context: str
    settings: Dict[str, Any]
    turns: List[Dict]
    questions: List[Dict]
    clusters: List[Dict]
    annotated: bool
    drafted: bool
    refined: bool
    answer_map: Dict[str, str]
    status: str
    next_action: str
    trace: List[Dict]
    errors: List[str]
    llm_mode: str


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(state: AgentState, thought: str, action: str, action_input: Any, observation: str) -> List[Dict]:
    trace = list(state.get("trace") or [])
    trace.append(
        {
            "step": len(trace) + 1,
            "time": _now(),
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        }
    )
    return trace


def _settings(state: AgentState) -> Dict[str, Any]:
    defaults = {
        "cluster_threshold": config.CLUSTER_THRESHOLD,
        "suggest_answers": True,
        "use_memory": True,
        "max_refine_rounds": config.MAX_REFINE_ROUNDS,
    }
    defaults.update(state.get("settings") or {})
    return defaults


# ------------------------------------------------------------------ 节点
@observability.observe_node("perceive")
def perceive(state: AgentState) -> Dict[str, Any]:
    """感知层：把用户上传/粘贴的原始对话变成结构化 Turn。"""
    turns = run_tool(
        "parse_chat_input",
        file_name=state.get("file_name", ""),
        file_bytes=state.get("file_bytes"),
        pasted_text=state.get("pasted_text", ""),
    )
    observation = f"识别到 {len(turns)} 条有效消息" if turns else "未识别到任何消息，需要用户重新提供对话"
    return {
        "turns": turns,
        "status": "perceived" if turns else "empty_input",
        "trace": _log(state, "先判断输入形态（文件/粘贴），再统一解析成消息序列", "parse_chat_input", {"file_name": state.get("file_name", ""), "has_text": bool(state.get("pasted_text"))}, observation),
    }


@observability.observe_node("plan")
def plan(state: AgentState) -> Dict[str, Any]:
    """决策层：根据 State 判断下一步该调用哪个工具。"""
    action, thought = decide_next(state)
    return {"next_action": action, "trace": _log(state, thought, "plan", {"state_keys": sorted(k for k, v in state.items() if v)}, f"决定下一步：{action}")}


def decide_next(state: AgentState) -> tuple[str, str]:
    if state.get("status") == "empty_input" or not state.get("turns"):
        return "clarify", "输入为空，无法继续，需要提示用户重新上传或粘贴对话"
    if not state.get("questions"):
        return "extract", "已有消息但还没有问题清单，下一步应抽取客户问题"
    if not state.get("clusters"):
        return "cluster", "已有关键问题，下一步需要做语义归并，把重复问法合并"
    if not state.get("annotated"):
        return "annotate", "已有问题簇，下一步计算频次占比与优先级并排序"
    settings = _settings(state)
    if settings.get("suggest_answers") and not state.get("drafted"):
        return "draft", "问题清单已成型，下一步检索历史记忆并生成建议回答"
    if _quality_low(state) and not state.get("refined"):
        return "refine", "发现大量问题簇只有 1 条，说明归并可能过细，尝试放宽阈值再合并一次"
    return "review", "问题清单质量达标，输出表格交给人工填写回答"


def _quality_low(state: AgentState) -> bool:
    clusters = state.get("clusters") or []
    questions = state.get("questions") or []
    if len(clusters) < 4 or not questions:
        return False
    singletons = sum(1 for c in clusters if int(c.get("count", 0)) <= 1)
    return singletons / len(clusters) > 0.7 and len(questions) >= 8


@observability.observe_node("extract")
def extract(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    questions = run_tool("extract_questions", turns=state.get("turns", []), business_context=state.get("business_context", ""), llm=llm)
    observation = f"抽取到 {len(questions)} 条客户诉求（模式：{llm.mode_label}）"
    if not questions:
        observation = "未抽取到问题，可能对话里没有客户提问"
    return {
        "questions": questions,
        "status": "extracted" if questions else "no_question",
        "llm_mode": llm.mode_label,
        "trace": _log(state, "逐条判断哪些消息是客户诉求，并把口语改写成可读问题", "extract_questions", {"turn_count": len(state.get("turns", []))}, observation),
    }


@observability.observe_node("cluster")
def cluster(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    settings = _settings(state)
    clusters = run_tool(
        "cluster_questions",
        questions=state.get("questions", []),
        threshold=float(settings["cluster_threshold"]),
        llm=llm,
        business_context=state.get("business_context", ""),
    )
    observation = f"合并为 {len(clusters)} 个问题簇，最大簇 {max((c['count'] for c in clusters), default=1)} 条同义问法"
    return {
        "clusters": clusters,
        "status": "clustered",
        "trace": _log(state, "对语义相同/诉求相同的问题做聚类，并选出规范问法", "cluster_questions", {"threshold": settings["cluster_threshold"]}, observation),
    }


@observability.observe_node("annotate")
def annotate(state: AgentState) -> Dict[str, Any]:
    clusters = run_tool("annotate_clusters", clusters=state.get("clusters", []), total_questions=len(state.get("questions", [])))
    top = clusters[0] if clusters else {}
    observation = f"标注完成，P0 问题 {sum(1 for c in clusters if c.get('priority') == 'P0')} 个；最高优先级：{top.get('question', '无')}"
    return {
        "clusters": clusters,
        "annotated": True,
        "status": "annotated",
        "trace": _log(state, "按出现频次、问题类型、紧急词计算优先级并排序", "annotate_clusters", {"cluster_count": len(clusters)}, observation),
    }


@observability.observe_node("draft")
def draft(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    settings = _settings(state)
    clusters = [dict(c) for c in state.get("clusters", [])]
    hits_total = 0
    for item in clusters:
        history: List[Dict] = []
        if settings.get("use_memory"):
            history = run_tool("retrieve_similar_answers", question=item.get("question", ""), k=3)
            history = [h for h in history if h.get("score", 0) >= 0.45]
            hits_total += len(history)
        item["memory_hits"] = history
        item["suggested_answer"] = run_tool(
            "draft_answer",
            cluster=item,
            history=history,
            business_context=state.get("business_context", ""),
            llm=llm,
        )
    observation = f"为 {len(clusters)} 个问题生成建议回答，命中历史记忆 {hits_total} 次"
    return {
        "clusters": clusters,
        "drafted": True,
        "status": "drafted",
        "trace": _log(state, "先检索历史已答问题，再结合业务背景生成可直接修改的话术草稿", "retrieve_similar_answers → draft_answer", {"use_memory": settings.get("use_memory")}, observation),
    }


@observability.observe_node("refine")
def refine(state: AgentState) -> Dict[str, Any]:
    settings = _settings(state)
    threshold = max(0.3, float(settings["cluster_threshold"]) * 0.72)
    clusters = run_tool(
        "cluster_questions",
        questions=state.get("questions", []),
        threshold=threshold,
        llm=get_llm(),
        business_context=state.get("business_context", ""),
    )
    clusters = run_tool("annotate_clusters", clusters=clusters, total_questions=len(state.get("questions", [])))
    return {
        "clusters": clusters,
        "refined": True,
        "annotated": True,
        "status": "refined",
        "trace": _log(state, "归并结果过于碎片化，主动降低相似度阈值再合并一轮", "cluster_questions", {"threshold": threshold}, f"二次合并后剩 {len(clusters)} 个问题簇"),
    }


@observability.observe_node("review")
def review(state: AgentState) -> Dict[str, Any]:
    clusters = state.get("clusters", [])
    return {
        "status": "awaiting_human",
        "trace": _log(state, "生成可编辑表格，等待人工确认与填写回答", "export_rows", {"rows": len(clusters)}, f"已输出 {len(clusters)} 行待回答问题，等待人工确认"),
    }


@observability.observe_node("clarify")
def clarify(state: AgentState) -> Dict[str, Any]:
    errors = list(state.get("errors") or []) + ["未解析到有效对话，请检查文件格式或重新粘贴内容"]
    return {
        "status": "empty_input",
        "errors": errors,
        "trace": _log(state, "输入不可用，终止流程并提示用户", "clarify", {}, "已向用户返回错误提示"),
    }


@observability.observe_node("finalize")
def finalize(state: AgentState) -> Dict[str, Any]:
    """人工确认后：合并用户填写的回答，写入 SQLite + Chroma。"""
    answer_map = state.get("answer_map") or {}
    clusters = [dict(c) for c in state.get("clusters", [])]
    for item in clusters:
        answer = str(answer_map.get(item.get("cluster_id", ""), "") or "").strip()
        if answer:
            item["answer"] = answer
    meta = {
        "file_name": state.get("file_name", ""),
        "turn_count": len(state.get("turns", [])),
        "question_count": len(state.get("questions", [])),
        "cluster_count": len(clusters),
        "status": "done",
        "llm_mode": state.get("llm_mode", "offline"),
    }
    saved = run_tool("save_to_memory", session_id=state.get("session_id", ""), clusters=clusters, meta=meta)
    observability.write_local_trace(state.get("session_id", "unknown"), state.get("trace", []), extra={"event": "finalize"})
    return {
        "clusters": clusters,
        "status": "done",
        "trace": _log(state, "人工回答已确认，写入长期记忆，下次遇到同类问题可直接复用口径", "save_to_memory", {"answered": saved["answered"]}, f"已入库 {saved['saved']} 条问题，其中 {saved['answered']} 条带人工回答"),
    }


# ------------------------------------------------------------------ 组装
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("perceive", perceive)
    builder.add_node("plan", plan)
    builder.add_node("extract", extract)
    builder.add_node("cluster", cluster)
    builder.add_node("annotate", annotate)
    builder.add_node("draft", draft)
    builder.add_node("refine", refine)
    builder.add_node("review", review)
    builder.add_node("clarify", clarify)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("perceive")
    builder.add_edge("perceive", "plan")
    builder.add_conditional_edges(
        "plan",
        lambda state: decide_next(state)[0],
        {
            "extract": "extract",
            "cluster": "cluster",
            "annotate": "annotate",
            "draft": "draft",
            "refine": "refine",
            "review": "review",
            "clarify": "clarify",
        },
    )
    for worker in ("extract", "cluster", "annotate", "draft", "refine"):
        builder.add_edge(worker, "plan")
    builder.add_edge("review", "finalize")
    builder.add_edge("clarify", END)
    builder.add_edge("finalize", END)

    # 人工确认环节：在 finalize 前中断，等用户填写完回答再恢复
    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["finalize"])


GRAPH = build_graph()


def _config(session_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 60}


def _to_result(state: Dict[str, Any]) -> AgentResult:
    clusters = [Cluster(**c) for c in (state.get("clusters") or [])]
    questions = state.get("questions") or []
    return AgentResult(
        session_id=state.get("session_id", ""),
        status=state.get("status", "unknown"),
        turn_count=len(state.get("turns") or []),
        question_count=len(questions),
        cluster_count=len(clusters),
        merge_rate=round(1 - len(clusters) / len(questions), 4) if questions else 0.0,
        llm_mode=state.get("llm_mode", "offline"),
        clusters=clusters,
        trace=state.get("trace") or [],
        errors=state.get("errors") or [],
    )


def run_agent(
    file_bytes: Optional[bytes] = None,
    file_name: str = "",
    pasted_text: str = "",
    business_context: str = "",
    settings: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> AgentResult:
    """启动一次完整分析：解析 → 抽取 → 归并 → 标注 → 建议回答 →（中断）等待人工填写。"""
    sid = session_id or new_session_id()
    state: AgentState = {
        "session_id": sid,
        "file_name": file_name,
        "file_bytes": file_bytes,
        "pasted_text": pasted_text,
        "business_context": business_context,
        "settings": settings or {},
        "trace": [],
        "errors": [],
        "answer_map": {},
        "llm_mode": get_llm().mode_label,
    }
    GRAPH.invoke(state, _config(sid))
    final_state = GRAPH.get_state(_config(sid)).values
    result = _to_result(final_state)

    memory.save_session(
        session_id=sid,
        file_name=file_name,
        turn_count=result.turn_count,
        question_count=result.question_count,
        cluster_count=result.cluster_count,
        status=result.status,
        llm_mode=result.llm_mode,
        meta={"business_context": business_context, "settings": settings or {}},
    )
    observability.write_local_trace(sid, result.trace, extra={"event": "run"})
    observability.flush()
    return result


def submit_answers(session_id: str, answers: Dict[str, str]) -> AgentResult:
    """人工填写回答后恢复图执行，进入 finalize 节点完成入库。"""
    cfg = _config(session_id)
    snapshot = GRAPH.get_state(cfg)
    if not snapshot.values:
        raise ValueError(f"会话 {session_id} 不存在或已结束")
    GRAPH.update_state(cfg, {"answer_map": answers, "status": "reviewed"})
    GRAPH.invoke(None, cfg)
    result = _to_result(GRAPH.get_state(cfg).values)
    observability.flush()
    return result


def get_state(session_id: str) -> Dict[str, Any]:
    return dict(GRAPH.get_state(_config(session_id)).values or {})

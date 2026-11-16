"""核心数据结构：贯穿 解析 → 抽取 → 归并 → 标注 → 人工回答 → 入库 全链路。"""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

QTYPES = ("咨询", "需求", "故障", "投诉")
CATEGORIES = ("售后退款", "价格费用", "物流配送", "功能使用", "账号权限", "技术故障", "发票合同", "其他")


class Turn(BaseModel):
    """一条群聊消息。"""

    idx: int = 0
    speaker: str = "客户"
    timestamp: str = ""
    text: str = ""
    source: str = "paste"


class RawQuestion(BaseModel):
    """从对话中抽取出的单条问题（未归并）。"""

    qid: str = ""
    text: str = ""
    qtype: str = "咨询"
    speaker: str = ""
    timestamp: str = ""
    source: str = ""
    turn_idx: int = -1


class Cluster(BaseModel):
    """归并后的问题簇，是最终表格的一行。"""

    cluster_id: str = ""
    question: str = ""
    category: str = "其他"
    qtype: str = "咨询"
    count: int = 0
    ratio: float = 0.0
    variants: List[str] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    priority: str = "P2"
    priority_score: float = 0.0
    tags: List[str] = Field(default_factory=list)
    quote: str = ""
    suggested_answer: str = ""
    answer: str = ""
    evidence: List[str] = Field(default_factory=list)


class SessionMeta(BaseModel):
    session_id: str = ""
    created_at: str = ""
    file_name: str = ""
    turn_count: int = 0
    question_count: int = 0
    cluster_count: int = 0
    status: str = "running"
    llm_mode: str = "offline"


class AgentResult(BaseModel):
    session_id: str = ""
    status: str = "awaiting_human"
    turn_count: int = 0
    question_count: int = 0
    cluster_count: int = 0
    merge_rate: float = 0.0
    llm_mode: str = "offline"
    clusters: List[Cluster] = Field(default_factory=list)
    trace: List[Dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

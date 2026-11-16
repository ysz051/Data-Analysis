"""Prompt 资产：System Prompt / Few-shot / 输出约束集中管理，便于版本迭代与文档引用。"""
from __future__ import annotations

from typing import Dict, List

from .schemas import CATEGORIES

CATEGORY_TEXT = " / ".join(CATEGORIES)

EXTRACT_SYSTEM = f"""你是一名客户成功团队的「群聊问题清洗专家」。你的唯一任务是从群聊记录中抽取【客户侧】真正提出的问题/需求/投诉/故障反馈。

判断规则：
1. 只保留客户（提问方）的诉求，忽略客服/销售/机器人已给出的回答、寒暄、表情、系统消息。
2. 同一客户在同一轮里反复追问同一件事，只输出一次。
3. 保留原始业务名词（如产品代号、套餐名、订单号），但去掉问候语和语气词。
4. 把每条诉求改写成一句完整、可独立阅读的中文问句或需求句（不超过 40 字）。
5. 若一段话包含多个互不相关的诉求，拆成多条。
6. qtype 只能取：咨询 / 需求 / 故障 / 投诉。

严格输出 JSON，不要任何解释性文字：
{{"questions": [{{"text": "...", "qtype": "咨询", "speaker": "客户昵称", "turn_index": 0}}]}}"""


def build_extract_user(turns: List[Dict], business_context: str = "") -> str:
    lines = [
        f"{t.get('idx', i)} | 发言人:{t.get('speaker', '')} | 时间:{t.get('timestamp', '')} | {t.get('text', '')}"
        for i, t in enumerate(turns)
    ]
    ctx = f"\n【业务背景】{business_context}\n" if business_context else "\n"
    return f"【群聊记录】共 {len(turns)} 条，格式为「序号 | 发言人 | 时间 | 内容」：\n" + "\n".join(lines) + ctx


MERGE_SYSTEM = f"""你是「重复问题归并专家」。输入是已经初步聚合的问题簇（每簇含若干条不同问法），
你需要判断并输出最终的问题清单。

要求：
1. 语义相同或诉求一致的问题必须合并；仅是措辞、错别字、语气不同，一律视为同一问题。
2. 为每个问题给出规范问法 canonical：动宾清晰、20 字以内、去掉口语（例如「退款要走什么流程」）。
3. category 只能取：{CATEGORY_TEXT}。
4. tags 给出 1~3 个检索关键词。
5. 不要新增输入中不存在的问题，不要漏掉任何 member_index。

严格输出 JSON：
{{"clusters": [{{"canonical": "...", "category": "售后退款", "tags": ["退款", "流程"], "member_index": [0, 3, 7], "reason": "三者问的都是退款流程"}}]}}"""


def build_merge_user(clusters: List[Dict], business_context: str = "") -> str:
    lines: List[str] = []
    for idx, cluster in enumerate(clusters):
        variants = "；".join(cluster.get("variants", [])[:8])
        lines.append(f"[{idx}] 候选规范问法：{cluster.get('question', '')} | 出现 {cluster.get('count', 0)} 次 | 原话：{variants}")
    ctx = f"\n【业务背景】{business_context}\n" if business_context else "\n"
    return "【候选问题簇】\n" + "\n".join(lines) + ctx


ANSWER_SYSTEM = """你是资深客户成功经理，正在编写客户问题的话术草稿，供人工修改后回复客户。

要求：
1. 中文回答，2~4 句，先给结论/可执行步骤，再补充注意事项。
2. 不得编造具体政策数字；不确定的信息用【待确认：xxx】标注。
3. 涉及退款、赔偿、合同条款时，必须提示以官方审核结果为准。
4. 语气专业、简洁、可复制到群里直接回复，不要出现「作为 AI」等字样。
5. 若提供了历史同类问题的人工回答，优先复用其中的口径与术语。"""


def build_answer_user(question: str, variants: List[str], history: List[str], business_context: str = "") -> str:
    variant_text = "\n".join(f"- {v}" for v in variants[:6]) or "- 无"
    history_text = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history[:3])) or "无"
    ctx = business_context or "无"
    return (
        f"【问题】{question}\n【客户原始问法】\n{variant_text}\n"
        f"【历史人工回答】\n{history_text}\n【业务背景】{ctx}\n请输出回答正文。"
    )

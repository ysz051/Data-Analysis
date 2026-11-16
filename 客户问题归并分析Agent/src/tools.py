"""Agent 工具集：8 个可独立调用、可被 LLM function-calling 选择的工具。

约定：每个工具都是「纯函数 + 显式入参 + 可序列化出参」，
便于在 LangGraph 节点中被调度，也便于单测与评测脚本直接调用。
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import memory
from .embeddings import cosine, embed_texts, normalize
from .llm import LLMClient
from .parsers import parse_bytes, parse_text
from .schemas import Cluster, RawQuestion, Turn

# ---------------------------------------------------------------- 规则词典
QUESTION_HINTS = (
    "?", "？", "怎么", "如何", "为什么", "为啥", "能否", "是否", "有没有", "多久", "多长时间",
    "几天", "几日", "多少天", "多少钱", "什么", "哪", "多少", "支持", "行不行", "可以吗", "吗", "呢", "能不能",
)
REQUIRE_HINTS = ("希望", "建议", "需要", "要求", "想要", "最好能", "麻烦", "增加", "新增", "优化", "改进", "支持一下")
ANSWER_HINTS = ("您好", "亲，", "已经为您", "已经帮您", "请您", "可以尝试", "我们这边", "收到", "好的呢", "感谢反馈")
URGENT_HINTS = ("急", "尽快", "马上", "催", "投诉", "差评", "赔偿", "严重", "无法", "不能用", "故障", "崩", "退款")
PROBLEM_HINTS = (
    "报错", "失败", "打不开", "崩", "卡住", "卡顿", "不能用", "无法", "收不到", "异常",
    "闪退", "转圈", "错误", "不支持", "没反应", "没到账", "没更新", "空的", "空白", "不行",
    "没收到", "没发货", "没通过", "过期", "失效", "不对", "有bug", "出问题", "坏了",
    "有问题", "不显示", "打不出来", "不好使", "用不了", "出错", "投诉", "太差",
)

CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("售后退款", ("退款", "退货", "换货", "售后", "保修", "维修", "退款进度", "退单")),
    ("价格费用", ("价格", "费用", "收费", "多少钱", "报价", "优惠", "折扣", "续费", "套餐", "性价比", "贵")),
    ("物流配送", ("发货", "物流", "快递", "配送", "到货", "运费", "签收", "多久到")),
    ("功能使用", ("怎么用", "如何使用", "功能", "教程", "操作", "设置", "导出", "导入", "批量", "模板")),
    ("账号权限", ("账号", "登录", "注册", "密码", "权限", "子账号", "成员", "授权", "实名")),
    ("技术故障", ("报错", "无法使用", "打不开", "崩溃", "卡顿", "失败", "异常", "闪退", "加载", "接口", "故障")),
    ("发票合同", ("发票", "开票", "合同", "盖章", "对公", "付款", "转账", "税号")),
]

ANSWER_TEMPLATES = {
    "售后退款": "已收到您的退款诉求。请在群内补充订单号与下单时间，我们会核对订单状态后按流程提交退款，退款到账时间以支付渠道为准。【待确认：具体到账时效】",
    "价格费用": "关于费用问题，我们的报价按实际使用规模分档计价。请告知您的账号/订单，我们会在群内同步对应的价格明细与优惠政策。",
    "物流配送": "关于发货与物流，正常情况下订单会在 48 小时内发出。请提供订单号，我们立即为您查询最新物流节点。",
    "功能使用": "该功能支持在后台按步骤操作：进入对应模块 → 选择目标数据 → 执行操作。若需要，我们可以在群内发一份图文操作指引。【待确认：具体菜单位置】",
    "账号权限": "账号相关问题需要先确认身份。请提供注册手机号后四位，我们会核查账号状态并协助重置权限。",
    "技术故障": "已记录该故障现象。请补充出现时间、操作路径和报错截图，我们同步技术同学排查，有结论第一时间在群内反馈。",
    "发票合同": "发票/合同支持对公办理。请提供开票信息（抬头、税号、金额）与收件邮箱，我们会在 3 个工作日内处理。【待确认：开票周期】",
    "其他": "已收到您的问题，我们会尽快核实并在群内回复。若方便，请补充更具体的场景信息，便于我们准确定位。",
}

TYPE_WEIGHT = {"投诉": 1.0, "故障": 0.9, "需求": 0.55, "咨询": 0.35}

# 主题级话术（比分类级更贴近真实业务，离线模式也能给出可用草稿）
TOPIC_ANSWERS: Dict[str, Dict[str, str]] = {
    "退款与售后": {
        "issue": "退款可在「订单详情 → 申请退款」提交，审核通过后按原支付渠道退回。到账时间以支付渠道为准，一般为 1~7 个工作日。【待确认：具体到账时效】",
        "ask": "我们的售后政策支持 7 天内无理由退款、质量问题免费换修。具体以订单页面展示的售后规则为准。",
    },
    "发票与合同": {
        "issue": "开票周期异常多为开票信息不完整导致。请补充抬头、税号与金额，我们会在 1~3 个工作日内重新开具。【待确认：加急开票可行性】",
        "ask": "发票支持电子普票与增值税专票，请提供抬头、税号、金额与接收邮箱；合同可走对公盖章流程。",
    },
    "登录与账号安全": {
        "issue": "登录异常请先确认账号是否被锁定。请提供注册手机号后四位与报错截图，我们协助重置密码或更换验证方式，紧急情况可临时开放人工验证。",
        "ask": "目前支持手机号 + 验证码、密码登录，微信扫码登录已在规划中。【待确认：上线时间】",
    },
    "价格与报价": {
        "issue": "如对账单或续费金额有疑问，请提供账号与订单号，我们核对后给出费用明细与处理方案。",
        "ask": "报价按账号数与功能模块组合分档，可拆分购买。请告知使用人数与所需模块，我们在群内提供正式报价单与优惠方案。",
    },
    "数据导出": {
        "issue": "导出失败通常是数据量过大或筛选条件为空导致。请缩小时间范围后重试，仍失败请提供报错截图，我们协助后台导出。",
        "ask": "导出前请先设置筛选条件，再点击「导出」；支持 Excel 与 CSV，单次上限 5 万行。【待确认：导出上限】",
    },
    "数据导入与模板": {
        "issue": "导入报错请核对模板列名与数据格式，常见原因是日期格式或必填列为空。",
        "ask": "批量导入可在「数据管理 → 导入」下载标准模板，按模板填写后上传，支持 xlsx / csv。",
    },
    "附件上传": {
        "issue": "若提示格式不支持，请确认文件扩展名与实际内容一致；仍失败可压缩后重试，或由我们在后台代传。",
        "ask": "附件支持 xlsx、csv、pdf、png 等格式，单个文件不超过 20M，超出请拆分上传。",
    },
    "物流与发货": {
        "issue": "物流停滞一般是快递揽收延迟。请提供订单号，我们立即向仓库核实并同步最新节点与预计到达时间。",
        "ask": "默认顺丰/京东发货，江浙沪次日达，其他地区 2~3 天，偏远地区 3~5 天；支持指定快递（到付）。",
    },
    "子账号与权限": {
        "issue": "子账号权限异常请提供主账号与子账号信息，我们核对角色配置后调整。",
        "ask": "子账号支持按角色分配权限，可在「设置 → 成员与权限」新增，并将数据范围限制为本人 / 本部门 / 全部。",
    },
}


# ---------------------------------------------------------------- 工具 1：解析
def parse_chat_input(file_name: str = "", file_bytes: Optional[bytes] = None, pasted_text: str = "") -> List[Dict]:
    """把多源输入统一解析成 [{idx, speaker, timestamp, text, source}]。"""
    turns = []
    if file_bytes:
        turns = parse_bytes(file_name or "upload.txt", file_bytes)
    if pasted_text.strip():
        turns.extend(parse_text(pasted_text, source="paste"))
        for i, turn in enumerate(turns):
            turn.idx = i
    return [t.model_dump() for t in turns]


# ---------------------------------------------------------------- 工具 2：抽取
def _looks_like_answer(text: str) -> bool:
    if len(text) > 80 and "？" not in text and "?" not in text:
        return True
    return any(hint in text for hint in ANSWER_HINTS) and "？" not in text


def _classify_type(text: str) -> str:
    if any(k in text for k in ("投诉", "差评", "太差", "垃圾", "生气", "赔偿", "骗")):
        return "投诉"
    if any(
        k in text
        for k in (
            "报错", "无法", "打不开", "崩", "失败", "不能用", "闪退", "卡", "异常",
            "收不到", "没反应", "坏了", "不对", "有bug", "出问题", "有问题", "出错", "不显示",
        )
    ):
        return "故障"
    if any(k in text for k in REQUIRE_HINTS):
        return "需求"
    return "咨询"


def _is_question(text: str) -> bool:
    if len(text) < 4 or _looks_like_answer(text):
        return False
    # 故障描述往往不带问号（如「验证码根本收不到」），也算客户诉求
    return any(hint in text for hint in QUESTION_HINTS + REQUIRE_HINTS + PROBLEM_HINTS)


def _split_multi(text: str) -> List[str]:
    """只有一句话里出现两个及以上问号时才拆分，避免把陈述句尾巴切成碎片。"""
    if text.count("？") + text.count("?") < 2:
        return [text.strip()]
    parts = [p.strip(" ，,、") for p in re.split(r"[？?]+", text) if p.strip(" ，,、")]
    heads = [p for p in parts if len(p) >= 4 and any(h in p for h in QUESTION_HINTS + REQUIRE_HINTS + PROBLEM_HINTS)]
    return heads if len(heads) > 1 else [text.strip()]


def _polish(text: str) -> str:
    cleaned = re.sub(r"^(请问一下|请问|想问下|想问|你好|您好|麻烦问下|麻烦)[，,\s]*", "", text).strip()
    cleaned = re.sub(r"[。！!]+$", "", cleaned)
    return cleaned or text.strip()


# 问题描述词：从「对象 + 故障描述」里剥离出问题对象（如「按钮坏了怎么办」→「按钮」）
ISSUE_PHRASES = (
    "出问题了", "出问题", "不能用了", "不能用", "用不了", "不好使", "打不开", "加载不出来",
    "没反应", "有bug", "bug", "坏了", "坏掉", "不对", "有误", "错了", "错误", "异常",
    "报错", "失败", "有问题", "不显示", "打不出来", "怎么办", "怎么弄", "怎么处理",
    "如何处理", "怎么解决", "如何解决", "为什么", "咋回事", "怎么回事", "如何", "怎么",
)
_ISSUE_SORTED = tuple(sorted(ISSUE_PHRASES, key=len, reverse=True))


def extract_subject(text: str) -> str:
    """从一句问题里提取「出问题的对象」，提取不到返回空串。

    「按钮坏了怎么办」→ 按钮；「商品信息不对」→ 商品信息；「页面出问题」→ 页面。
    """
    cleaned = _polish(text)
    removed = False
    for phrase in _ISSUE_SORTED:
        if phrase in cleaned:
            cleaned = cleaned.replace(phrase, "")
            removed = True
    cleaned = re.sub(r"[，。！？、,.!?；;：:\s]", "", cleaned)
    cleaned = cleaned.strip("的了呢吗啊呀吧着")
    if not removed or not (2 <= len(cleaned) <= 8):
        return ""
    return cleaned


AGENT_NAME_HINTS = ("客服", "助手", "官方", "机器人", "bot", "售后", "销售", "顾问", "支持", "运营")


def infer_agent_speakers(turns: List[Dict]) -> set:
    """从昵称推断哪些发言人是「我方客服」，用于过滤掉回答类消息。"""
    speakers = {str(t.get("speaker", "")).strip() for t in turns if str(t.get("speaker", "")).strip()}
    agents = {s for s in speakers if any(hint in s.lower() for hint in AGENT_NAME_HINTS)}
    if not speakers or len(agents) == len(speakers):
        return set()
    return agents


def extract_questions(turns: List[Dict], business_context: str = "", llm: Optional[LLMClient] = None) -> List[Dict]:
    """规则抽取（保底）+ LLM 语义抽取（增强），最终按原文去重。"""
    llm = llm or LLMClient()
    collected: List[Dict] = []

    llm_questions = llm.extract_questions(turns, business_context) if llm.enabled else None
    if llm_questions:
        for item in llm_questions:
            text = _polish(str(item.get("text", "")).strip())
            if len(text) < 4:
                continue
            turn_idx = item.get("turn_index", item.get("turn_idx", -1))
            try:
                turn_idx = int(turn_idx)
            except (TypeError, ValueError):
                turn_idx = -1
            turn = turns[turn_idx] if 0 <= turn_idx < len(turns) else {}
            collected.append(
                {
                    "text": text,
                    "qtype": item.get("qtype") if item.get("qtype") in TYPE_WEIGHT else _classify_type(text),
                    "speaker": item.get("speaker") or turn.get("speaker", "客户"),
                    "timestamp": item.get("timestamp") or turn.get("timestamp", ""),
                    "source": turn.get("source", "paste"),
                    "turn_idx": turn_idx,
                }
            )

    if not collected:
        agent_speakers = infer_agent_speakers(turns)
        for turn in turns:
            if str(turn.get("speaker", "")) in agent_speakers:
                continue
            text = str(turn.get("text", "")).strip()
            if not _is_question(text):
                continue
            for piece in _split_multi(text):
                polished = _polish(piece)
                collected.append(
                    {
                        "text": polished,
                        "qtype": _classify_type(polished),
                        "speaker": turn.get("speaker", "客户"),
                        "timestamp": turn.get("timestamp", ""),
                        "source": turn.get("source", "paste"),
                        "turn_idx": turn.get("idx", -1),
                    }
                )

    # 同一人同一句只保留一次，避免刷屏放大频次
    seen = set()
    questions: List[RawQuestion] = []
    for item in collected:
        key = (normalize(item["text"]), item.get("speaker", ""))
        if key in seen:
            continue
        seen.add(key)
        questions.append(
            RawQuestion(
                qid=f"Q{len(questions) + 1:03d}",
                text=item["text"],
                qtype=item["qtype"],
                speaker=item.get("speaker", ""),
                timestamp=item.get("timestamp", ""),
                source=item.get("source", ""),
                turn_idx=int(item.get("turn_idx", -1)),
            )
        )
    return [q.model_dump() for q in questions]


# ---------------------------------------------------------------- 工具 3：归并
def _union_find_merge(questions: List[Dict], threshold: float) -> List[List[int]]:
    n = len(questions)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    norms = [normalize(q["text"]) for q in questions]
    vectors = embed_texts([q["text"] for q in questions])
    for i in range(n):
        for j in range(i + 1, n):
            a, b = norms[i], norms[j]
            if not a or not b:
                continue
            if a == b:
                union(i, j)
                continue
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if len(shorter) >= 6 and shorter in longer and len(shorter) / len(longer) >= 0.6:
                union(i, j)
                continue
            if cosine(vectors[i], vectors[j]) >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# 主题意图词典：同一主题 + 同一诉求性质（问题 / 咨询）即视为重复问题，
# 解决「字面差异大、语义相同」时纯相似度算法合并不到一起的问题。
TOPIC_RULES: List[Dict[str, Any]] = [
    {
        "topic": "退款与售后",
        "category": "售后退款",
        "keywords": ("退款", "退货", "退单", "退钱", "售后", "保修", "维修"),
        "issue": "退款申请入口与退款到账时效？",
        "ask": "退款 / 售后政策与操作流程是什么？",
    },
    {
        "topic": "发票与合同",
        "category": "发票合同",
        "keywords": ("发票", "开票", "税号", "专票", "普票", "合同", "盖章", "对公", "报销"),
        "issue": "发票开具异常 / 开票周期过长如何处理？",
        "ask": "发票如何开具、开票周期是多久？",
    },
    {
        "topic": "登录与账号安全",
        "category": "账号权限",
        "keywords": ("登录", "登陆", "验证码", "密码", "锁定", "502", "扫码"),
        "issue": "登录失败 / 验证码收不到如何排查？",
        "ask": "能否支持更便捷的登录方式（微信 / 扫码）？",
    },
    {
        "topic": "价格与报价",
        "category": "价格费用",
        "keywords": ("价格", "报价", "多少钱", "费用", "收费", "优惠", "折扣", "续费", "年费", "套餐", "贵"),
        "issue": "费用争议 / 续费异常如何处理？",
        "ask": "产品报价、套餐与优惠政策如何获取？",
    },
    {
        "topic": "数据导出",
        "category": "功能使用",
        "keywords": ("导出", "下载"),
        "issue": "导出失败 / 导出数据为空如何排查？",
        "ask": "数据导出功能怎么用、支持哪些格式？",
    },
    {
        "topic": "数据导入与模板",
        "category": "功能使用",
        "keywords": ("导入", "模板"),
        "issue": "批量导入报错如何处理？",
        "ask": "批量导入方式与模板在哪里下载？",
    },
    {
        "topic": "附件上传",
        "category": "功能使用",
        "keywords": ("上传", "附件", "文件大小", "格式支持", "文件格式"),
        "issue": "附件上传报错如何处理？",
        "ask": "附件上传支持哪些格式与大小限制？",
    },
    {
        "topic": "物流与发货",
        "category": "物流配送",
        "keywords": ("物流", "快递", "发货", "到货", "配送", "签收", "运费"),
        "issue": "物流停滞 / 未发货如何处理？",
        "ask": "发货方式与物流时效如何？",
    },
    {
        "topic": "子账号与权限",
        "category": "账号权限",
        "keywords": ("子账号", "权限", "成员", "部门数据", "授权"),
        "issue": "子账号权限配置异常如何处理？",
        "ask": "子账号与数据权限如何配置？",
    },
]

QTYPE_SEVERITY = {"投诉": 3, "故障": 2, "需求": 1, "咨询": 0}


def detect_topic(text: str) -> Optional[Dict[str, Any]]:
    """按词典顺序命中主题；顺序即优先级（更具体的主题放前面）。"""
    for rule in TOPIC_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            return rule
    return None


def _inherit_topics(questions: List[Dict]) -> List[Optional[Dict[str, Any]]]:
    """上下文继承：没有主题词的追问，沿用同一发言人前后 3 轮内的主题。"""
    topics = [detect_topic(q.get("text", "")) for q in questions]
    for i, q in enumerate(questions):
        if topics[i] is not None:
            continue
        turn_idx = int(q.get("turn_idx", -1))
        if turn_idx < 0:
            continue
        best_idx, best_dist = None, 99
        for j, other in enumerate(questions):
            if topics[j] is None or other.get("speaker") != q.get("speaker"):
                continue
            other_idx = int(other.get("turn_idx", -1))
            if other_idx < 0:
                continue
            dist = abs(turn_idx - other_idx)
            if dist <= 3 and dist < best_dist:
                best_idx, best_dist = j, dist
        if best_idx is not None:
            topics[i] = topics[best_idx]
    return topics


def _guess_category(text: str, qtype: str) -> str:
    for category, keywords in CATEGORY_RULES:
        if any(k in text for k in keywords):
            return category
    if qtype == "故障":
        return "技术故障"
    return "其他"


def _canonical(variants: Sequence[Dict]) -> Dict:
    """选代表问法：频次最高的归一化形式，长度适中者优先。"""
    counter: Dict[str, int] = {}
    for v in variants:
        counter[normalize(v["text"])] = counter.get(normalize(v["text"]), 0) + 1
    ranked = sorted(
        variants,
        key=lambda v: (counter.get(normalize(v["text"]), 0), -abs(len(v["text"]) - 16), len(v["text"])),
        reverse=True,
    )
    return ranked[0]


def cluster_questions(
    questions: List[Dict],
    threshold: float = 0.55,
    llm: Optional[LLMClient] = None,
    business_context: str = "",
) -> List[Dict]:
    """三级归并：主题意图（业务域）→ 问题对象（按钮/页面/商品信息…）→ 相似度兜底。"""
    llm = llm or LLMClient()
    if not questions:
        return []

    topics = _inherit_topics(questions)
    buckets: Dict[str, List[int]] = {}
    subject_buckets: Dict[str, List[int]] = {}
    leftovers: List[int] = []
    for i, q in enumerate(questions):
        topic = topics[i]
        if topic is not None:
            buckets.setdefault(topic["topic"], []).append(i)
            continue
        subject = extract_subject(q.get("text", ""))
        if subject:
            subject_buckets.setdefault(subject, []).append(i)
        else:
            leftovers.append(i)

    grouped: List[Tuple[List[int], Optional[Dict], str, str]] = []
    for key, indexes in buckets.items():
        variants = [questions[i] for i in indexes]
        problem_count = sum(1 for v in variants if v.get("qtype") in ("故障", "投诉"))
        kind = "issue" if problem_count * 2 >= len(variants) else "ask"
        rule = next(rule for rule in TOPIC_RULES if rule["topic"] == key)
        grouped.append((indexes, rule, kind, ""))

    # 同一「问题对象」归为一类，单条也保留成行，保证所有问题都能看到
    for subject, indexes in subject_buckets.items():
        grouped.append((indexes, None, "subject", subject))

    if leftovers:
        sub = [questions[i] for i in leftovers]
        for group in _union_find_merge(sub, threshold):
            grouped.append(([leftovers[k] for k in group], None, "similar", ""))

    candidates: List[Dict] = []
    for indexes, topic, kind, subject in grouped:
        variants = [questions[i] for i in indexes]
        rep = _canonical(variants)
        type_counter: Dict[str, int] = {}
        for v in variants:
            key = v.get("qtype", "咨询")
            type_counter[key] = type_counter.get(key, 0) + 1
        qtype = max(type_counter, key=lambda t: (type_counter[t], QTYPE_SEVERITY.get(t, 0)))
        if topic:
            question = topic["issue"] if kind == "issue" else topic["ask"]
            category = topic["category"]
            tags = [topic["topic"]]
        elif kind == "subject":
            question = f"{subject}相关问题"
            category = _guess_category(" ".join(v["text"] for v in variants), qtype)
            tags = [subject]
        else:
            question = rep["text"]
            category = _guess_category(" ".join(v["text"] for v in variants), qtype)
            tags = []
        candidates.append(
            {
                "cluster_id": f"C{len(candidates) + 1:02d}",
                "question": question,
                "category": category,
                "qtype": qtype,
                "kind": kind if kind != "subject" else "issue",
                "variants": [v["text"] for v in variants],
                "members": indexes,
                "count": len(variants),
                "speakers": sorted({v.get("speaker", "") for v in variants if v.get("speaker")}),
                "sources": sorted({v.get("source", "") for v in variants if v.get("source")}),
                "tags": tags,
                "first_seen": next((v.get("timestamp") for v in variants if v.get("timestamp")), ""),
                "last_seen": next((v.get("timestamp") for v in reversed(variants) if v.get("timestamp")), ""),
            }
        )
    candidates.sort(key=lambda c: c["count"], reverse=True)

    if llm.enabled:
        merged = llm.name_clusters(candidates, business_context)
        if merged:
            try:
                candidates = _apply_llm_merge(candidates, merged)
            except (KeyError, IndexError, TypeError, ValueError):
                pass

    clusters: List[Cluster] = []
    for c in candidates:
        clusters.append(
            Cluster(
                cluster_id=f"C{len(clusters) + 1:02d}",
                question=c["question"],
                category=c["category"],
                qtype=c["qtype"],
                count=c["count"],
                variants=c["variants"],
                speakers=c["speakers"],
                sources=c["sources"],
                first_seen=c.get("first_seen", ""),
                last_seen=c.get("last_seen", ""),
                tags=c.get("tags", []),
                quote=c["variants"][0] if c["variants"] else "",
            )
        )
    return [c.model_dump() for c in clusters]


def _apply_llm_merge(candidates: List[Dict], merged: List[Dict]) -> List[Dict]:
    """把 LLM 给出的 member_index 重新组装；校验覆盖完整性，失败则由调用方兜底。"""
    covered: List[int] = []
    rebuilt: List[Dict] = []
    for item in merged:
        indexes = [int(i) for i in item.get("member_index", [])]
        if not indexes or any(i < 0 or i >= len(candidates) for i in indexes):
            continue
        members: List[int] = []
        variants: List[str] = []
        for i in indexes:
            members.extend(candidates[i]["members"])
            variants.extend(candidates[i]["variants"])
        covered.extend(indexes)
        base = dict(candidates[indexes[0]])
        category = item.get("category") if item.get("category") in {c for c, _ in CATEGORY_RULES} | {"其他"} else base["category"]
        rebuilt.append(
            {
                **base,
                "question": _polish(str(item.get("canonical") or base["question"])),
                "category": category,
                "count": len(variants),
                "members": members,
                "variants": variants,
                "tags": [str(t) for t in item.get("tags", [])][:3],
            }
        )
    if len(set(covered)) != len(candidates):
        raise ValueError("LLM 归并结果未覆盖全部候选簇，回退规则归并")
    rebuilt.sort(key=lambda c: c["count"], reverse=True)
    return rebuilt


# ---------------------------------------------------------------- 工具 4：标注
def annotate_clusters(clusters: List[Dict], total_questions: int = 0) -> List[Dict]:
    """计算频次占比 / 优先级 / 标签，并补齐可读字段。"""
    total = max(1, total_questions or sum(c.get("count", 0) for c in clusters))
    for c in clusters:
        count = int(c.get("count", 0))
        c["ratio"] = round(count / total, 4)
        text = " ".join([c.get("question", "")] + c.get("variants", []))
        urgency = any(k in text for k in URGENT_HINTS)
        score = 0.6 * min(count / 4.0, 1.0) + 0.25 * TYPE_WEIGHT.get(c.get("qtype", "咨询"), 0.35) + 0.15 * (1.0 if urgency else 0.0)
        c["priority_score"] = round(min(score, 1.0), 3)
        c["priority"] = "P0" if score >= 0.72 else ("P1" if score >= 0.5 else "P2")
        if not c.get("tags"):
            c["tags"] = [c.get("category", "其他")] + (["高频"] if count >= 3 else [])
        c.setdefault("evidence", [])
        if not c["evidence"]:
            c["evidence"] = c.get("variants", [])[:3]
        if not c.get("quote"):
            c["quote"] = c["evidence"][0] if c["evidence"] else c.get("question", "")
    clusters.sort(key=lambda c: (c.get("priority_score", 0), c.get("count", 0)), reverse=True)
    return clusters


# ---------------------------------------------------------------- 工具 5：记忆检索
def retrieve_similar_answers(question: str, k: int = 3) -> List[Dict]:
    """从长期记忆（Chroma）检索历史已答问题，用于复用口径。"""
    return memory.search_similar(question, k=k)


# ---------------------------------------------------------------- 工具 6：草拟回答
def draft_answer(
    cluster: Dict,
    history: List[Dict],
    business_context: str = "",
    llm: Optional[LLMClient] = None,
) -> str:
    """生成建议回答：LLM 优先，离线时用分类模板 + 历史口径。"""
    llm = llm or LLMClient()
    history_answers = [h.get("answer", "") for h in history if h.get("answer")]
    if llm.enabled:
        text = llm.draft_answer(cluster.get("question", ""), cluster.get("variants", []), history_answers, business_context)
        if text and text.strip():
            return text.strip()

    category = cluster.get("category", "其他")
    topic = (cluster.get("tags") or [""])[0]
    kind = cluster.get("kind") or ("issue" if cluster.get("qtype") in ("故障", "投诉") else "ask")
    if topic and topic not in TOPIC_ANSWERS and str(cluster.get("question", "")).endswith("相关问题"):
        template = (
            f"已收到关于「{topic}」的问题。请补充出现页面、操作步骤与截图，"
            "我们同步排查后第一时间在群内回复处理进展。"
        )
    else:
        template = (TOPIC_ANSWERS.get(topic) or {}).get(kind) or ANSWER_TEMPLATES.get(category, ANSWER_TEMPLATES["其他"])
    if history_answers:
        return f"{template}\n（历史口径参考：{history_answers[0][:80]}）"
    return template


# ---------------------------------------------------------------- 工具 7：表格导出
def export_rows(clusters: List[Dict], answers: Optional[Dict[str, str]] = None) -> List[Dict]:
    """转换成用户可见/可编辑的表格结构（中文表头）。"""
    answers = answers or {}
    rows: List[Dict] = []
    for i, c in enumerate(clusters, 1):
        rows.append(
            {
                "序号": i,
                "问题（已合并）": c.get("question", ""),
                "出现次数": int(c.get("count", 0)),
                "占比": f"{round(float(c.get('ratio', 0)) * 100, 1)}%",
                "优先级": c.get("priority", "P2"),
                "问题类型": c.get("qtype", "咨询"),
                "分类": c.get("category", "其他"),
                "客户原声示例": c.get("quote", ""),
                "其他问法": " / ".join([v for v in c.get("variants", []) if v != c.get("question", "")][:3]),
                "建议回答": c.get("suggested_answer", ""),
                "我的回答": answers.get(c.get("cluster_id", ""), c.get("answer", "")),
                "cluster_id": c.get("cluster_id", ""),
            }
        )
    return rows


# ---------------------------------------------------------------- 工具 8：入库
def save_to_memory(session_id: str, clusters: List[Dict], meta: Optional[Dict] = None) -> Dict:
    """会话与问答写入 SQLite + Chroma，形成可复用的长期记忆。"""
    memory.save_clusters(session_id, clusters)
    memory.save_session(
        session_id=session_id,
        file_name=(meta or {}).get("file_name", ""),
        turn_count=int((meta or {}).get("turn_count", 0)),
        question_count=int((meta or {}).get("question_count", 0)),
        cluster_count=int((meta or {}).get("cluster_count", len(clusters))),
        status=(meta or {}).get("status", "done"),
        llm_mode=(meta or {}).get("llm_mode", "offline"),
        meta=meta or {},
    )
    answered = sum(1 for c in clusters if str(c.get("answer", "")).strip())
    return {"saved": len(clusters), "answered": answered, "session_id": session_id}


# ---------------------------------------------------------------- 工具注册表
TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "parse_chat_input",
        "description": "解析 Excel/CSV/Word/纯文本群聊记录，统一为 Turn 列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string", "description": "上传文件名，用于判断格式"},
                "file_bytes": {"type": "string", "format": "binary", "description": "文件二进制内容（与 pasted_text 二选一）"},
                "pasted_text": {"type": "string", "description": "直接粘贴的对话文本"},
            },
            "required": [],
        },
        "output_schema": {"type": "array", "items": {"$ref": "Turn", "properties": {"idx": {"type": "integer"}, "speaker": {"type": "string"}, "timestamp": {"type": "string"}, "text": {"type": "string"}, "source": {"type": "string"}}}},
        "example": '{"file_name": "group_chat.csv", "file_bytes": "<binary>"}',
    },
    {
        "name": "extract_questions",
        "description": "从 Turn 列表中抽取客户问题/需求/投诉，输出结构化 RawQuestion。",
        "input_schema": {
            "type": "object",
            "properties": {
                "turns": {"type": "array", "items": {"type": "object"}},
                "business_context": {"type": "string", "description": "业务背景，提升抽取准确率"},
            },
            "required": ["turns"],
        },
        "output_schema": {"type": "array", "items": {"properties": {"qid": {"type": "string"}, "text": {"type": "string"}, "qtype": {"type": "string", "enum": ["咨询", "需求", "故障", "投诉"]}}}},
        "example": '{"turns": [{"idx": 0, "speaker": "张三", "text": "退款怎么弄？"}], "business_context": "SaaS 财务软件"}',
    },
    {
        "name": "cluster_questions",
        "description": "对问题做语义聚类与归并，生成规范问法与问题簇。",
        "input_schema": {
            "type": "object",
            "properties": {
                "questions": {"type": "array", "items": {"type": "object"}},
                "threshold": {"type": "number", "minimum": 0.3, "maximum": 0.95, "description": "相似度阈值，越高越不容易合并"},
                "business_context": {"type": "string"},
            },
            "required": ["questions"],
        },
        "output_schema": {"type": "array", "items": {"properties": {"cluster_id": {"type": "string"}, "question": {"type": "string"}, "count": {"type": "integer"}, "variants": {"type": "array"}}}},
        "example": '{"questions": [{"text": "退款怎么弄？"}, {"text": "退款流程是什么？"}], "threshold": 0.55}',
    },
    {
        "name": "annotate_clusters",
        "description": "计算占比、优先级（P0/P1/P2）、标签，并按优先级排序。",
        "input_schema": {
            "type": "object",
            "properties": {"clusters": {"type": "array"}, "total_questions": {"type": "integer"}},
            "required": ["clusters"],
        },
        "output_schema": {"type": "array", "items": {"properties": {"priority": {"type": "string"}, "priority_score": {"type": "number"}, "ratio": {"type": "number"}}}},
        "example": '{"clusters": [{"cluster_id": "C01", "count": 4}], "total_questions": 12}',
    },
    {
        "name": "retrieve_similar_answers",
        "description": "从向量库检索历史已答的相似问题，返回 answer 与相似度。",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "k": {"type": "integer", "default": 3}},
            "required": ["question"],
        },
        "output_schema": {"type": "array", "items": {"properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "score": {"type": "number"}}}},
        "example": '{"question": "退款流程是什么？", "k": 3}',
    },
    {
        "name": "draft_answer",
        "description": "为问题簇生成建议回答草稿（LLM 优先，离线用模板）。",
        "input_schema": {
            "type": "object",
            "properties": {"cluster": {"type": "object"}, "history": {"type": "array"}, "business_context": {"type": "string"}},
            "required": ["cluster"],
        },
        "output_schema": {"type": "string", "description": "中文回答草稿"},
        "example": '{"cluster": {"question": "退款流程是什么？", "category": "售后退款"}, "history": []}',
    },
    {
        "name": "export_rows",
        "description": "把问题簇转换成中文表头、可直接展示/编辑/导出的行数据。",
        "input_schema": {"type": "object", "properties": {"clusters": {"type": "array"}, "answers": {"type": "object"}}, "required": ["clusters"]},
        "output_schema": {"type": "array", "items": {"properties": {"序号": {"type": "integer"}, "问题（已合并）": {"type": "string"}, "出现次数": {"type": "integer"}, "我的回答": {"type": "string"}}}},
        "example": '{"clusters": [{"cluster_id": "C01"}], "answers": {"C01": "..."}}',
    },
    {
        "name": "save_to_memory",
        "description": "把本次会话与问答写入 SQLite 与 Chroma，供后续检索复用。",
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}, "clusters": {"type": "array"}, "meta": {"type": "object"}},
            "required": ["session_id", "clusters"],
        },
        "output_schema": {"type": "object", "properties": {"saved": {"type": "integer"}, "answered": {"type": "integer"}}},
        "example": '{"session_id": "s-20261116", "clusters": [...], "meta": {"status": "done"}}',
    },
]

_TOOL_MAP = {
    "parse_chat_input": parse_chat_input,
    "extract_questions": extract_questions,
    "cluster_questions": cluster_questions,
    "annotate_clusters": annotate_clusters,
    "retrieve_similar_answers": retrieve_similar_answers,
    "draft_answer": draft_answer,
    "export_rows": export_rows,
    "save_to_memory": save_to_memory,
}


def run_tool(name: str, **kwargs) -> Any:
    """统一工具调度入口：Agent 的 Action 都走这里，便于 Trace 与权限管控。"""
    if name not in _TOOL_MAP:
        raise KeyError(f"未知工具：{name}")
    return _TOOL_MAP[name](**kwargs)


def new_session_id() -> str:
    return f"S{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"

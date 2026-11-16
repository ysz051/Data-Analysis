"""多源输入解析：Excel / CSV / Word / 纯文本 → 统一 Turn 列表。

不依赖任何外部服务，保证 Agent 的「感知层」在离线环境也可用。
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Iterable, List, Optional

import pandas as pd
from docx import Document

from .schemas import Turn

# 内容列 / 说话人列 / 时间列 的候选名（中英文、常见导出模板）
TEXT_COLS = ["内容", "消息", "聊天内容", "对话", "文本", "问题", "content", "message", "text", "msg", "question"]
SPEAKER_COLS = ["发送人", "昵称", "用户", "客户", "说话人", "姓名", "sender", "speaker", "user", "name", "from", "author"]
TIME_COLS = ["时间", "发送时间", "日期", "time", "timestamp", "date", "datetime", "created_at"]

SYSTEM_NOISE = (
    "[图片]", "[表情]", "[语音]", "[视频]", "[文件]", "[链接]", "[动画表情]", "[位置]",
    "撤回了一条消息", "加入了群聊", "邀请", "领取了", "拍了拍", "以下为新消息", "红包",
    "@所有人", "该消息已过期", "开启了群聊", "修改群名为",
)

_HEADER_TS = re.compile(
    r"^(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<speaker>.{1,24}?)\s*$"
)
# 单行「时间 昵称：内容」格式（如 2026-11-15 09:00:12 王女士：退款怎么操作）
_LEAD_TIME = re.compile(
    r"^(?P<time>\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}:\d{2}(?::\d{2})?)\s+"
)
_SPEAKER_LINE = re.compile(r"^(?P<speaker>[^:：\s]{1,20})\s*[:：]\s*(?P<text>.+)$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_noise(text: str) -> bool:
    if not text:
        return True
    return any(token in text for token in SYSTEM_NOISE) and len(text) <= 20


def _pick_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _turns_from_frame(df: pd.DataFrame, source: str, offset: int = 0) -> List[Turn]:
    df = df.dropna(how="all")
    if df.empty:
        return []
    text_col = _pick_column(df.columns, TEXT_COLS)
    if text_col is None:
        # 退化策略：取文本长度均值最大的列作为内容列
        best, best_len = df.columns[0], -1
        for col in df.columns:
            avg = df[col].astype(str).str.len().mean()
            if avg > best_len:
                best, best_len = col, avg
        text_col = best
    speaker_col = _pick_column(df.columns, SPEAKER_COLS)
    time_col = _pick_column(df.columns, TIME_COLS)

    turns: List[Turn] = []
    for _, row in df.iterrows():
        text = _clean(row.get(text_col, ""))
        if _is_noise(text):
            continue
        speaker = _clean(row.get(speaker_col, "")) if speaker_col else "客户"
        ts = _clean(row.get(time_col, "")) if time_col else ""
        turns.append(
            Turn(
                idx=offset + len(turns),
                speaker=speaker or "客户",
                timestamp=ts,
                text=text,
                source=source,
            )
        )
    return turns


def parse_text(text: str, source: str = "paste") -> List[Turn]:
    """解析「昵称：内容」或微信「时间 昵称 / 内容」两种常见粘贴格式。"""
    turns: List[Turn] = []
    speaker, ts, pending_header = "客户", "", False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _HEADER_TS.match(line)
        if header:
            rest = line[header.end("time"):].strip()
            if "：" not in rest and ":" not in rest:
                speaker, ts, pending_header = header.group("speaker").strip(), header.group("time"), True
                continue
        lead = _LEAD_TIME.match(line)
        if lead:
            ts = lead.group("time")
            line = line[lead.end():].strip()
            pending_header = False
        matched = _SPEAKER_LINE.match(line)
        if matched and len(matched.group("speaker")) <= 20:
            speaker = matched.group("speaker").strip()
            line = matched.group("text").strip()
        elif not pending_header:
            speaker = "客户"
        pending_header = False
        if _is_noise(line):
            continue
        turns.append(Turn(idx=len(turns), speaker=speaker or "客户", timestamp=ts, text=line, source=source))
    return turns


def _parse_docx(data: bytes, source: str, offset: int = 0) -> List[Turn]:
    doc = Document(io.BytesIO(data))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [_clean(c.text) for c in row.cells if _clean(c.text)]
            if cells:
                lines.append("：".join(cells[:2]) if len(cells) >= 2 else cells[0])
    return parse_text("\n".join(lines), source=source)[offset:]


def parse_bytes(file_name: str, data: bytes) -> List[Turn]:
    """按扩展名分派解析器，返回按时间顺序排列的 Turn 列表。"""
    name = (file_name or "").lower()
    source = file_name or "upload"

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        book = pd.read_excel(io.BytesIO(data), sheet_name=None)
        turns: List[Turn] = []
        for sheet, frame in book.items():
            turns.extend(_turns_from_frame(frame, source=f"{source}#{sheet}", offset=len(turns)))
        return turns

    if name.endswith(".csv"):
        for encoding in ("utf-8-sig", "gb18030", "utf-8"):
            try:
                frame = pd.read_csv(io.BytesIO(data), encoding=encoding)
                return _turns_from_frame(frame, source=source)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        return parse_text(data.decode("utf-8", errors="ignore"), source=source)

    if name.endswith((".docx", ".doc")):
        try:
            return _parse_docx(data, source=source)
        except (zipfile.BadZipFile, ValueError, KeyError):
            return parse_text(data.decode("utf-8", errors="ignore"), source=source)

    return parse_text(data.decode("utf-8", errors="ignore"), source=source)

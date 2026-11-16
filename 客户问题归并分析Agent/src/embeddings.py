"""本地确定性 Embedding：字符 n-gram 哈希向量。

DeepSeek 未提供 embedding 接口，且课程环境可能无外网，
因此这里用「无需模型下载、可复现」的哈希向量作为向量库编码器：
- 语义能力弱于 BGE/text-embedding-3，但对短问题文本的近似去重足够；
- 若接入在线 embedding，只需替换本文件的 embed_texts 实现。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

DIM = 256
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）()【】\[\]…,.!?;:~`\-_/\\]+")
# 口语同义归一，提升相似问题命中率
SYNONYMS = {
    "咋": "怎么", "咋样": "怎么样", "多少钱": "价格", "价钱": "价格", "费用": "价格",
    "可不可以": "可以", "能不能": "可以", "能否": "可以", "行不行": "可以",
    "多长时间": "多久", "几天": "多久", "什么时候": "何时",
    "登录不上": "无法登录", "登不上": "无法登录", "打不开": "无法打开",
    "崩溃": "无法使用", "报错": "无法使用", "用不了": "无法使用",
}
POLITE = ("请问一下", "请问", "想问下", "想问", "你好", "您好", "麻烦问下", "麻烦", "谢谢", "辛苦了", "在吗")


def normalize(text: str) -> str:
    s = (text or "").lower().strip()
    for polite in POLITE:
        s = s.replace(polite, "")
    for src, dst in SYNONYMS.items():
        s = s.replace(src, dst)
    s = _PUNCT.sub("", s)
    return s


def _ngrams(text: str, ns: Sequence[int] = (2, 3)) -> List[str]:
    out: List[str] = []
    for n in ns:
        out.extend(text[i : i + n] for i in range(max(0, len(text) - n + 1)))
    return out or ([text] if text else [])


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        vec = [0.0] * DIM
        for gram in _ngrams(normalize(text)):
            digest = hashlib.md5(gram.encode("utf-8")).hexdigest()
            slot = int(digest[:8], 16) % DIM
            sign = 1.0 if int(digest[8:9], 16) % 2 == 0 else -1.0
            vec[slot] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HashEmbeddingFunction(EmbeddingFunction):
    """Chroma 自定义 EmbeddingFunction（chromadb>=0.4 规范）。"""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - 遵循 chromadb 签名
        return embed_texts(list(input))

    @staticmethod
    def name() -> str:
        return "hash-ngram-256"

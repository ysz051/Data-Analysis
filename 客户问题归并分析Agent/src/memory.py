"""记忆层。

- 短期记忆：当前会话的对话/问题/簇，保存在 LangGraph 的 checkpoint（MemorySaver）与 SQLite 会话表；
- 长期记忆：SQLite 存结构化问答，Chroma 存语义向量，支持「新问题→检索历史已答问题」复用口径。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

from . import config
from .embeddings import HashEmbeddingFunction

_lock = threading.Lock()
_chroma_client = None
_collection = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    file_name TEXT,
    turn_count INTEGER DEFAULT 0,
    question_count INTEGER DEFAULT 0,
    cluster_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    llm_mode TEXT DEFAULT 'offline',
    meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    cluster_id TEXT,
    question TEXT,
    category TEXT,
    qtype TEXT,
    count INTEGER,
    priority TEXT,
    priority_score REAL,
    variants TEXT,
    tags TEXT,
    suggested_answer TEXT,
    answer TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_clusters_session ON clusters(session_id);
"""


def _conn() -> sqlite3.Connection:
    config.MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.MEMORY_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock, _conn() as conn:
        conn.executescript(SCHEMA)


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        import chromadb

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=HashEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def save_session(
    session_id: str,
    file_name: str = "",
    turn_count: int = 0,
    question_count: int = 0,
    cluster_count: int = 0,
    status: str = "running",
    llm_mode: str = "offline",
    meta: Optional[Dict] = None,
) -> None:
    init_db()
    with _lock, _conn() as conn:
        conn.execute(
            """INSERT INTO sessions(session_id, created_at, file_name, turn_count, question_count,
                                    cluster_count, status, llm_mode, meta)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                 turn_count=excluded.turn_count, question_count=excluded.question_count,
                 cluster_count=excluded.cluster_count, status=excluded.status,
                 llm_mode=excluded.llm_mode, meta=excluded.meta""",
            (
                session_id,
                datetime.now().isoformat(timespec="seconds"),
                file_name,
                turn_count,
                question_count,
                cluster_count,
                status,
                llm_mode,
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )


def save_clusters(session_id: str, clusters: List[Dict]) -> None:
    """写入结构化问答；已有回答的问题才会进入向量库供后续复用。"""
    init_db()
    now = datetime.now().isoformat(timespec="seconds")
    with _lock, _conn() as conn:
        for c in clusters:
            conn.execute(
                """INSERT INTO clusters(id, session_id, cluster_id, question, category, qtype, count,
                                        priority, priority_score, variants, tags, suggested_answer, answer, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     count=excluded.count, priority=excluded.priority, priority_score=excluded.priority_score,
                     variants=excluded.variants, tags=excluded.tags,
                     suggested_answer=excluded.suggested_answer, answer=excluded.answer,
                     updated_at=excluded.updated_at""",
                (
                    f"{session_id}:{c.get('cluster_id', '')}",
                    session_id,
                    c.get("cluster_id", ""),
                    c.get("question", ""),
                    c.get("category", ""),
                    c.get("qtype", ""),
                    int(c.get("count", 0)),
                    c.get("priority", ""),
                    float(c.get("priority_score", 0.0)),
                    json.dumps(c.get("variants", []), ensure_ascii=False),
                    json.dumps(c.get("tags", []), ensure_ascii=False),
                    c.get("suggested_answer", ""),
                    c.get("answer", ""),
                    now,
                ),
            )

    answered = [c for c in clusters if str(c.get("answer", "")).strip()]
    if not answered:
        return
    collection = _get_collection()
    collection.upsert(
        ids=[f"{session_id}:{c['cluster_id']}" for c in answered],
        documents=[c["question"] for c in answered],
        metadatas=[
            {
                "question": c.get("question", ""),
                "answer": c.get("answer", ""),
                "category": c.get("category", ""),
                "session_id": session_id,
            }
            for c in answered
        ],
    )


def search_similar(question: str, k: int = 3) -> List[Dict]:
    """长期记忆检索：返回历史已答问题（带相似度分数），无记录时返回空列表。"""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return []
        result = collection.query(query_texts=[question], n_results=min(k, collection.count()))
    except Exception:  # noqa: BLE001 - 向量库异常不应阻断主流程
        return []

    hits: List[Dict] = []
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "question": (meta or {}).get("question", doc),
                "answer": (meta or {}).get("answer", ""),
                "category": (meta or {}).get("category", ""),
                "score": round(1.0 - float(dist), 4),
            }
        )
    return hits


def list_sessions(limit: int = 20) -> List[Dict]:
    init_db()
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT session_id, created_at, file_name, turn_count, question_count, cluster_count, status, llm_mode "
            "FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_session_clusters(session_id: str) -> List[Dict]:
    init_db()
    with _lock, _conn() as conn:
        rows = conn.execute("SELECT * FROM clusters WHERE session_id=? ORDER BY count DESC", (session_id,)).fetchall()
    return [dict(row) for row in rows]


def memory_stats() -> Dict[str, int]:
    init_db()
    with _lock, _conn() as conn:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    try:
        vectors = _get_collection().count()
    except Exception:  # noqa: BLE001
        vectors = 0
    return {"sessions": sessions, "clusters": clusters, "vectors": vectors}

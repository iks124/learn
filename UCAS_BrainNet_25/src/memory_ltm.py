from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .data import get_last_session_name

@dataclass
class RagConfig:
    embedding_model: str
    top_k: int
    max_ctx_chars: int
    index_fields: List[str]

def obs_to_string(obs: dict) -> str:
    lines = []
    for person, items in obs.items():
        lines.append(f"{person}:")
        for text, source in items:
            lines.append(f"- {text} ({source})")
    return "\n".join(lines)


class LongTermRAG:
    """
    对每个 sample 单独建一个小索引（最简单、最稳，适合课程作业与case可解释性）。
    如果你追求速度，可改成全量索引 + sample过滤。
    """
    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg.embedding_model)

    def _collect_docs(self, sample: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        conv = sample.get("conversation", {})
        docs: List[Tuple[str, Dict[str, Any]]] = []

        # 收集各 session 的 summary/observation（这非常像“长期痕迹”：压缩后的情景记忆）
        sessions = []
        for k in conv.keys():
            if k.startswith("session_") and k.count("_") == 1:
                sessions.append(k)
        sessions = sorted(sessions, key=lambda s: int(s.split("_")[1]))

        for s in sessions:
            meta = {"session": s}
            data = conv.get(f"{s}_date_time", "")
            if "session_summary" in self.cfg.index_fields:
                #summ = conv.get(f"{s}_summary", None)
                summ = sample["session_summary"].get(f"{s}_summary", None)  # 注意 summary 可能在 sample 顶层
                if summ:
                    docs.append((f"[{s}_summary] data_time: {data} summary: {summ}", {**meta, "field": "summary"}))
            if "observation" in self.cfg.index_fields:
                obs = sample["observation"].get(f"{s}_observation", None)
                if obs:
                    obs_str = obs_to_string(obs)
                    docs.append(( f"[{s}_observation] data_time: {data} observation: {obs_str}", {**meta, "field": "observation"} ))

        return docs

    def retrieve(self, sample: Dict[str, Any], query: str) -> Tuple[str, List[Dict[str, Any]]]:
        docs = self._collect_docs(sample)
        if not docs:
            return "", []

        texts = [d[0] for d in docs]
        metas = [d[1] for d in docs]

        emb = self.embedder.encode(texts, normalize_embeddings=True)
        q = self.embedder.encode([query], normalize_embeddings=True)

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb.astype(np.float32))

        scores, idxs = index.search(q.astype(np.float32), k=min(self.cfg.top_k, len(texts)))

        chosen = []
        ctx_parts = []
        total = 0
        for rank, (i, sc) in enumerate(zip(idxs[0].tolist(), scores[0].tolist())):
            if i < 0:
                continue
            t = texts[i]
            if total + len(t) > self.cfg.max_ctx_chars:
                break
            ctx_parts.append(t)
            total += len(t)
            chosen.append({**metas[i], "rank": rank, "score": float(sc), "text": t})

        ctx = "\n".join(ctx_parts)
        return ctx, chosen

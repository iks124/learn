from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple, Iterable

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_qa_items(sample: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    # LoCoMo: sample["qa"] = list of {question, answer, category, evidence}
    for i, qa in enumerate(sample.get("qa", [])):
        yield {
            "qa_idx": i,
            "question": qa.get("question", ""),
            "answer": qa.get("adversarial_answer", qa.get("answer", "")),
            "category": qa.get("category", ""),
            "evidence": qa.get("evidence", None),
        }

def flatten_conversation(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return list of turns sorted by session index then dia_id order (assumes stored order is chronological).
    Each turn: {session_id, speaker, dia_id, text, img_url, blip_caption}
    """
    conv = sample.get("conversation", {})
    # keys like: speaker_a, speaker_b, session_1, session_1_date_time, ...
    sessions = []
    for k in conv.keys():
        if k.startswith("session_") and not k.endswith("_date_time") and not k.endswith("_observation") and not k.endswith("_summary"):
            sessions.append(k)
    def session_num(s: str) -> int:
        return int(s.split("_")[1])

    turns_all: List[Dict[str, Any]] = []
    for sname in sorted(sessions, key=session_num):
        turns = conv.get(sname, [])
        data_time = conv.get(f"{sname}_date_time", [])
        for t in turns:
            turns_all.append({
                "data_time": data_time,
                "session": sname,
                "speaker": t.get("speaker", ""),
                "dia_id": t.get("dia_id", ""),
                "text": t.get("text", t.get("content", "")) or "",
                "img_url": t.get("img_url", None),
                "blip_caption": t.get("blip_caption", None),
            })
    return turns_all

def get_last_session_name(sample: Dict[str, Any]) -> str | None:
    conv = sample.get("conversation", {})
    sessions = []
    for k in conv.keys():
        if k.startswith("session_") and not k.endswith("_date_time") and not k.endswith("_observation") and not k.endswith("_summary"):
            sessions.append(k)
    if not sessions:
        return None
    return sorted(sessions, key=lambda s: int(s.split("_")[1]))[-1]

def get_session_aux(sample: Dict[str, Any], session_name: str) -> Tuple[str | None, str | None]:
    conv = sample.get("conversation", {})
    obs = conv.get(f"{session_name}_observation", None)
    summ = conv.get(f"{session_name}_summary", None)
    # 可能是 list 或 str，统一转 str
    def to_str(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x
        return json.dumps(x, ensure_ascii=False)
    return to_str(obs), to_str(summ)

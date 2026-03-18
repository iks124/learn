from __future__ import annotations
from typing import Any, Dict, List
from .data import flatten_conversation, get_last_session_name, get_session_aux

def _turn_to_str(t: Dict[str, Any]) -> str:
    # 如含图像caption，也纳入（LoCoMo里图片转文字本质是“感知痕迹”）
    extra = ""
    if t.get("blip_caption"):
        extra = f" [image_caption: {t['blip_caption']}]"
    return f"({t['data_time']} {t['session']}) {t['speaker']}: {t['text']}{extra}".strip()

def build_concat_context(sample: Dict[str, Any], max_chars: int) -> str:
    turns = flatten_conversation(sample)
    text = "\n".join(_turn_to_str(t) for t in turns)
    return text[-max_chars:] if len(text) > max_chars else text

def build_stm_context(sample: Dict[str, Any], last_k_turns: int, include_last_session_summary: bool, max_chars: int) -> str:
    turns = flatten_conversation(sample)
    recent = turns[-last_k_turns:] if last_k_turns > 0 else turns
    l0 = "\n".join(_turn_to_str(t) for t in recent)

    l1 = ""
    if include_last_session_summary:
        last_sess = get_last_session_name(sample)
        if last_sess:
            obs, summ = get_session_aux(sample, last_sess)
            parts = []
            if summ:
                parts.append(f"[last_session_summary] {summ}")
            if obs:
                parts.append(f"[last_session_observation] {obs}")
            if parts:
                l1 = "\n".join(parts)

    ctx = "\n".join([x for x in [l1, l0] if x])
    return ctx[-max_chars:] if len(ctx) > max_chars else ctx

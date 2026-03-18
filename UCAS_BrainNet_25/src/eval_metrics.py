from __future__ import annotations
import re
from typing import Tuple

def normalize(s: str, lower: bool = True) -> str:
    s = s.strip()
    if lower:
        s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, gold: str, lower: bool = True) -> int:
    return int(normalize(pred, lower) == normalize(gold, lower))

def llm_as_judge(
    judge_llm,
    question: str,
    pred: str,
    gold: str,
) -> int:
    system = "你是一个严格的问答评测员。"
    user = f"""【问题】
{question}

【参考答案】
{gold}

【模型回答】
{pred}

判断模型回答是否与参考答案在语义上等价。
只回答 YES 或 NO。
"""
    out = judge_llm.generate(system, user)
    out = out.strip().upper()
    return 1 if out.startswith("YES") else 0


def token_f1(pred: str, gold: str, lower: bool = True) -> float:
    p = normalize(pred, lower).split()
    g = normalize(gold, lower).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    num_same = 0
    for w in g:
        if common.get(w, 0) > 0:
            num_same += 1
            common[w] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

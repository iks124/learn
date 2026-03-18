from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict
import yaml
from tqdm import tqdm

from src.data import load_json, iter_qa_items
from src.prompts import SYSTEM, build_user_prompt
from src.llm_vllm import VLLMClient, VLLMConfig
from src.memory_stm import build_concat_context, build_stm_context
from src.memory_ltm import LongTermRAG, RagConfig
from src.eval_metrics import exact_match, token_f1, llm_as_judge

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dump_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_group(samples, group_name: str, cfg: Dict[str, Any], llm: VLLMClient, rag: LongTermRAG | None):
    out_dir = os.path.join(cfg["out_dir"], group_name)
    ensure_dir(out_dir)
    rows = []
    em_sum, f1_sum, n = 0, 0.0, 0

    for sample in tqdm(samples, desc=f"Running {group_name}"):
        sample_id = sample.get("sample_id", "")
        for qa in iter_qa_items(sample):
            q = qa["question"]
            gold = qa["answer"]

            # build context by group
            retrieved_ctx = ""
            retrieved_meta = []

            if group_name == "concat":
                context = build_concat_context(sample, max_chars=int(cfg["concat_max_chars"]))

            elif group_name == "stm":
                sc = cfg["stm"]
                context = build_stm_context(
                    sample,
                    last_k_turns=int(sc["last_k_turns"]),
                    include_last_session_summary=bool(sc["include_last_session_summary"]),
                    max_chars=int(sc["max_chars"]),
                )

            elif group_name == "ltm":
                assert rag is not None
                retrieved_ctx, retrieved_meta = rag.retrieve(sample, q)
                context = retrieved_ctx 

            elif group_name == "hybrid":
                assert rag is not None
                # STM 部分
                sc = cfg["stm"]
                stm_ctx = build_stm_context(
                    sample,
                    last_k_turns=int(sc["last_k_turns"]),
                    include_last_session_summary=bool(sc["include_last_session_summary"]),
                    max_chars=int(sc["max_chars"]),
                )
                # LTM 部分（用hybrid配置的top_k / max_chars，需要临时覆盖一下）
                # 简化：直接用 cfg["hybrid"] 重新初始化一个小rag配置（同embedding模型）
                hc = cfg["hybrid"]
                rag_h = LongTermRAG(RagConfig(
                    embedding_model=cfg["ltm"]["embedding_model"],
                    top_k=int(hc["top_k"]),
                    max_ctx_chars=int(cfg["ltm"]["max_ctx_chars"]),
                    index_fields=list(cfg["ltm"]["index_fields"]),
                ))
                retrieved_ctx, retrieved_meta = rag_h.retrieve(sample, q)

                # 混合拼接：先放长期检索（像“回忆再激活”），再放短期窗口（像“当前工作态”）
                mix = "\n".join([x for x in ["[LTM_Retrieved]\n" + retrieved_ctx if retrieved_ctx else "",
                                             "[STM]\n" + stm_ctx if stm_ctx else ""] if x])
                # 再裁剪总长
                max_chars = int(hc["max_chars"])
                context = mix[-max_chars:] if len(mix) > max_chars else mix

            else:
                raise ValueError(group_name)

            user_prompt = build_user_prompt(q, context)
            pred = llm.generate(SYSTEM, user_prompt)

            lower = bool(cfg["eval"]["lower"])
            em = llm_as_judge(
                judge_llm=llm,# 生成和judge用同一个模型
                question=q,
                pred=pred,
                gold=gold,
                )

            row = {
                "group": group_name,
                "sample_id": sample_id,
                "qa_idx": qa["qa_idx"],
                "category": qa["category"],
                "question": q,
                "gold": gold,
                "pred": pred,
                "exact_match": em,
                "evidence": qa["evidence"],
                "context": context,
                "retrieved": retrieved_meta,   # LTM/Hybrid 的检索细节
            }
            rows.append(row)

            em_sum += em
            n += 1

    metrics = {
        "group": group_name,
        "n": n,
        "exact_match": (em_sum / n) if n else 0.0,
    }

    dump_jsonl(os.path.join(out_dir, "results.jsonl"), rows)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    samples = load_json(cfg["data_path"])
    # 有些版本是 dict 包一层；这里做兼容
    if isinstance(samples, dict) and "data" in samples:
        samples = samples["data"]

    llm_cfg = VLLMConfig(**cfg["vllm"])
    llm = VLLMClient(llm_cfg)

    rag = LongTermRAG(RagConfig(
        embedding_model=cfg["ltm"]["embedding_model"],
        top_k=int(cfg["ltm"]["top_k"]),
        max_ctx_chars=int(cfg["ltm"]["max_ctx_chars"]),
        index_fields=list(cfg["ltm"]["index_fields"]),
    ))

    ensure_dir(cfg["out_dir"])

    all_metrics = []
    for group in ["hybrid", "concat", "stm", "ltm"]:
    #for group in ["hybrid"]:
        m = run_group(samples, group, cfg, llm, rag)
        all_metrics.append(m)

    with open(os.path.join(cfg["out_dir"], "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

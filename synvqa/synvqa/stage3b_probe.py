"""Stage 3b — Parametric-Gap Probe.

Only samples the target VLM does NOT already know are kept.

  1. Text-only probe of target_vlm, greedy, record predicted answer + logprob conf.
  2. Semantic-match judge (via judge_llm) decides correct vs incorrect.
  3. Gap decision: pass iff  (not correct)  OR  (correct but confidence < τ_conf).
  4. Optional: for samples that passed due to low confidence (but correct),
     re-probe with paraphrases — only treat as gap if ≥ 1 paraphrase is wrong/uncertain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.llm import LLMClient
from .models.vlm import VLMClient
from .utils.io import load_prompt, render_prompt
from .utils.json_parse import parse_strict_json
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_SMATCH = _ROOT / "prompts" / "semantic_answer_match.txt"
_PROMPT_PARAPH = _ROOT / "prompts" / "paraphrase_question.txt"


def run(
    *,
    samples: list[dict],
    config: dict[str, Any],
    target_vlm: VLMClient,
    judge_llm: LLMClient,
    generator_llm: LLMClient,
) -> list[dict]:
    log = get_logger("synvqa.stage3b")
    cfg = config["stage3b"]
    smatch_tmpl = load_prompt(_PROMPT_SMATCH)
    paraph_tmpl = load_prompt(_PROMPT_PARAPH)

    for sample in samples:
        if sample.get("reject_reason"):
            continue
        _probe_one(sample, cfg, target_vlm, judge_llm, generator_llm,
                   smatch_tmpl, paraph_tmpl)

    kept = sum(1 for s in samples if s.get("reject_reason") is None)
    log.info("stage 3b: %d/%d kept (gap samples)", kept, len(samples))
    return samples


def _semantic_match(judge: LLMClient, tmpl: str, question: str, gold: str,
                    predicted: str, prov: ProvenanceBuilder) -> dict:
    prompt = render_prompt(tmpl, question=question, gold=gold, predicted=predicted)
    try:
        resp = judge.complete(prompt, temperature=0.0, response_format_json=True)
        prov.model_call(role="judge", model=resp.model, prompt_text=prompt,
                        api_call_id=resp.api_call_id, extra={"usage": resp.usage})
        return parse_strict_json(resp.text)
    except Exception as e:
        return {"match": False, "reason": f"judge_error:{e}"}


def _probe_once(target: VLMClient, question: str, decode_cfg: dict
                ) -> tuple[str, float | None, list[str]]:
    try:
        r = target.generate_text(
            question,
            temperature=decode_cfg.get("temperature", 0.0),
            max_new_tokens=decode_cfg.get("max_new_tokens", 64),
            return_logprobs=decode_cfg.get("return_logprobs", True),
        )
        return r.text, r.logprob_confidence, []
    except Exception:
        return "", None, []


def _probe_one(
    sample: dict,
    cfg: dict,
    target: VLMClient,
    judge: LLMClient,
    generator: LLMClient,
    smatch_tmpl: str,
    paraph_tmpl: str,
) -> None:
    prov = ProvenanceBuilder(stage=32)
    q = sample["anchor_question"]
    gold = sample["answer"]
    decode_cfg = cfg["target_vlm_decode"]

    pred, conf, _ = _probe_once(target, q, decode_cfg)
    prov.log("probe", pred=pred, conf=conf)

    # Fallback if no logprobs: multi-sample consistency
    if conf is None:
        n = cfg.get("consistency_n_samples", 5)
        try:
            samples_ = target.sample_texts(q, n=n, temperature=0.7,
                                           max_new_tokens=decode_cfg.get("max_new_tokens", 64))
        except Exception:
            samples_ = []
        if samples_:
            top = max(set(samples_), key=samples_.count)
            conf = samples_.count(top) / len(samples_)
            pred = pred or top
        prov.log("consistency_fallback", conf=conf, n=len(samples_))

    match_out = _semantic_match(judge, smatch_tmpl, q, gold, pred, prov)
    correct = bool(match_out.get("match", False))
    tau = cfg["tau_confidence"]
    conf_val = 0.0 if conf is None else float(conf)

    is_gap = (not correct) or (conf_val < tau)

    # Ambiguous zone: correct + borderline confidence → paraphrase recheck.
    if (not is_gap) and cfg.get("paraphrase_recheck", True):
        # User spec: recheck correct+confident to guard against lucky hits.
        # We recheck when correct AND confidence is not clearly high.
        tau_high = min(0.9, tau + 0.2)
        if conf_val < tau_high:
            paraphrases = _paraphrase(generator, paraph_tmpl, q, cfg["n_paraphrases"], prov)
            ok_all = True
            pp_results = []
            for pp in paraphrases:
                p_text, p_conf, _ = _probe_once(target, pp, decode_cfg)
                p_match = _semantic_match(judge, smatch_tmpl, pp, gold, p_text, prov)
                p_ok = bool(p_match.get("match")) and (
                    p_conf is None or p_conf >= tau
                )
                pp_results.append({"paraphrase": pp, "pred": p_text,
                                   "conf": p_conf, "ok": p_ok})
                if not p_ok:
                    ok_all = False
            prov.log("paraphrase_recheck", results=pp_results)
            if not ok_all:
                is_gap = True

    sample["probe"] = {
        "predicted_answer": pred,
        "logprob_confidence": conf_val,
        "correct": correct,
        "decision": "pass" if is_gap else "fail",
    }
    if not is_gap:
        sample["reject_reason"] = "not_a_parametric_gap"
    sample.setdefault("provenance", {})["stage_3b"] = prov.finalize(
        correct=correct, conf=conf_val, is_gap=is_gap,
    )


def _paraphrase(generator: LLMClient, tmpl: str, q: str, n: int,
                prov: ProvenanceBuilder) -> list[str]:
    prompt = render_prompt(tmpl, question=q, n=n)
    try:
        resp = generator.complete(prompt, temperature=0.7, response_format_json=True)
        prov.model_call(role="generator", model=resp.model, prompt_text=prompt,
                        api_call_id=resp.api_call_id, extra={"usage": resp.usage})
        out = parse_strict_json(resp.text)
        pps = out.get("paraphrases", [])
        return [p for p in pps if isinstance(p, str) and p.strip()][:n]
    except Exception:
        return []

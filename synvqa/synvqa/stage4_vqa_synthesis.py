"""Stage 4 — VQA Synthesis.

Turns (anchor_question, answer, fact) into (caption, vqa_question, vqa_answer)
subject to three alignment constraints (visual sufficiency, non-leakage,
fact necessity). Abstract facts are diverted to text_aux_set.jsonl.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models.embeddings import EmbeddingClient
from .models.llm import LLMClient
from .utils.io import append_jsonl, load_prompt, render_prompt
from .utils.json_parse import parse_strict_json
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_VIZ = _ROOT / "prompts" / "visualizability_judge.txt"
_PROMPT_SYN = _ROOT / "prompts" / "vqa_synthesis.txt"
_PROMPT_ALN = _ROOT / "prompts" / "alignment_verifier.txt"


def run(
    *,
    samples: list[dict],
    output_dir: str | Path,
    config: dict[str, Any],
    generator_llm: LLMClient,
    judge_llm: LLMClient,
    embedder: EmbeddingClient,
) -> list[dict]:
    log = get_logger("synvqa.stage4")
    cfg = config["stage4"]
    viz_tmpl = load_prompt(_PROMPT_VIZ)
    syn_tmpl = load_prompt(_PROMPT_SYN)
    aln_tmpl = load_prompt(_PROMPT_ALN)

    aux_path = Path(output_dir) / "text_aux_set.jsonl"

    for sample in samples:
        if sample.get("reject_reason"):
            continue
        _synthesize(sample, cfg, generator_llm, judge_llm, embedder,
                    viz_tmpl, syn_tmpl, aln_tmpl, aux_path)

    kept = sum(1 for s in samples if s.get("reject_reason") is None
               and s.get("vqa_question"))
    log.info("stage 4: %d/%d kept", kept, len(samples))
    return samples


def _synthesize(
    sample: dict,
    cfg: dict,
    generator: LLMClient,
    judge: LLMClient,
    embedder: EmbeddingClient,
    viz_tmpl: str,
    syn_tmpl: str,
    aln_tmpl: str,
    aux_path: Path,
) -> None:
    prov = ProvenanceBuilder(stage=4)

    # Visualizability gate
    viz_prompt = render_prompt(viz_tmpl, fact=sample["fact"])
    try:
        vr = generator.complete(viz_prompt, temperature=0.0, response_format_json=True)
        prov.model_call(role="generator", model=vr.model, prompt_text=viz_prompt,
                        api_call_id=vr.api_call_id, extra={"usage": vr.usage})
        viz = parse_strict_json(vr.text)
    except Exception as e:
        sample["reject_reason"] = f"visualizability_failed:{e}"
        sample.setdefault("provenance", {})["stage_4"] = prov.finalize()
        return
    if not viz.get("visualizable"):
        sample["reject_reason"] = "abstract_fact"
        sample.setdefault("provenance", {})["stage_4"] = prov.finalize(viz=viz)
        append_jsonl(aux_path, {**sample, "abstract_reason": viz.get("reason")})
        return

    for attempt in range(cfg["max_retries"] + 1):
        syn_prompt = render_prompt(
            syn_tmpl,
            anchor_question=sample["anchor_question"],
            answer=sample["answer"],
            fact=sample["fact"],
        )
        try:
            sr = generator.complete(syn_prompt, temperature=0.7 if attempt else 0.5,
                                    response_format_json=True, max_tokens=1024)
            prov.model_call(role="generator", model=sr.model, prompt_text=syn_prompt,
                            api_call_id=sr.api_call_id, extra={"usage": sr.usage,
                                                              "attempt": attempt})
            syn = parse_strict_json(sr.text)
        except Exception as e:
            prov.log("synthesis_parse_error", error=str(e), attempt=attempt)
            continue

        caption = syn.get("caption", "")
        vqa_q = syn.get("vqa_question", "")
        vqa_a = syn.get("vqa_answer", "")
        anchor = syn.get("visual_anchor", "")

        # Fast local checks first.
        if _answer_leaks_in_caption(vqa_a, caption, embedder,
                                    cfg["leakage_check_mode"],
                                    cfg["leakage_embedding_threshold"]):
            prov.log("local_leakage", attempt=attempt)
            continue

        # Three-constraint judge.
        aln_prompt = render_prompt(
            aln_tmpl,
            caption=caption,
            vqa_question=vqa_q,
            vqa_answer=vqa_a,
            visual_anchor=anchor,
            fact=sample["fact"],
        )
        try:
            ar = judge.complete(aln_prompt, temperature=0.0, response_format_json=True)
            prov.model_call(role="judge", model=ar.model, prompt_text=aln_prompt,
                            api_call_id=ar.api_call_id, extra={"usage": ar.usage})
            aln = parse_strict_json(ar.text)
        except Exception as e:
            prov.log("alignment_judge_error", error=str(e), attempt=attempt)
            continue

        if aln.get("C1_visual_sufficiency") and aln.get("C2_no_leakage") \
                and aln.get("C3_fact_necessity") \
                and aln.get("C4_event_specificity", True):
            sample["caption"] = caption
            sample["vqa_question"] = vqa_q
            sample["vqa_answer"] = vqa_a
            sample["visual_anchor"] = anchor
            sample.setdefault("provenance", {})["stage_4"] = prov.finalize(
                attempts=attempt + 1, alignment=aln,
            )
            return
        prov.log("alignment_fail", attempt=attempt, result=aln)

    sample["reject_reason"] = "alignment_failed"
    sample.setdefault("provenance", {})["stage_4"] = prov.finalize(attempts=cfg["max_retries"] + 1)


def _answer_leaks_in_caption(answer: str, caption: str, embedder: EmbeddingClient,
                             mode: str, threshold: float) -> bool:
    if not answer or not caption:
        return False
    ans_l = answer.strip().lower()
    cap_l = caption.strip().lower()
    if "substring" in mode and ans_l and ans_l in cap_l:
        return True
    if "embedding" in mode:
        try:
            sim = embedder.similarity(answer, caption)
            if sim >= threshold:
                return True
        except Exception:
            pass
    return False

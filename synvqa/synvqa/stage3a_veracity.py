"""Stage 3a — Veracity Filter.

Confirms the fact is factually correct and well-sourced:
  - Groups sources by domain; requires ≥ k_independent distinct domains.
  - LLM-as-Judge (family different from generator) scores the claim.
  - Rejects if score < threshold OR too few independent high-reliability sources.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.llm import LLMClient
from .utils.io import load_prompt, render_prompt
from .utils.json_parse import parse_strict_json
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT = _ROOT / "prompts" / "veracity_judge.txt"


def run(
    *,
    samples: list[dict],
    config: dict[str, Any],
    judge_llm: LLMClient,
) -> list[dict]:
    log = get_logger("synvqa.stage3a")
    cfg = config["stage3a"]
    tmpl = load_prompt(_PROMPT)

    for sample in samples:
        if sample.get("reject_reason"):
            continue
        _veracity_check(sample, cfg, judge_llm, tmpl)

    kept = sum(1 for s in samples if s.get("reject_reason") is None)
    log.info("stage 3a: %d/%d kept", kept, len(samples))
    return samples


def _veracity_check(sample: dict, cfg: dict, judge: LLMClient, tmpl: str) -> None:
    prov = ProvenanceBuilder(stage=31)

    # Source independence grouping
    snips = sample.get("supporting_snippets", [])
    evidence = {e["url"]: e for e in sample.get("retrieved_evidence", [])}
    domains: dict[str, str] = {}  # domain → tier
    for s in snips:
        url = s.get("url")
        if url and url in evidence:
            domains[evidence[url]["domain"]] = evidence[url]["reliability_tier"]

    independent_high = sum(1 for t in domains.values() if t in ("high", "medium"))
    if independent_high < cfg["min_independent_sources"]:
        sample["reject_reason"] = "insufficient_independent_sources"
        sample["veracity"] = {"score": None, "n_independent_sources": independent_high,
                              "decision": "fail"}
        sample.setdefault("provenance", {})["stage_3a"] = prov.finalize(
            independent_high=independent_high,
        )
        return

    snippets_block = "\n".join(
        f"- [{evidence.get(s['url'], {}).get('reliability_tier', '?')}] "
        f"{s.get('url', '')}: {s.get('quote', '')}"
        for s in snips
    )
    prompt = render_prompt(tmpl, fact=sample["fact"], snippets=snippets_block)
    try:
        resp = judge.complete(prompt, temperature=0.0, response_format_json=True)
        prov.model_call(role="judge", model=resp.model, prompt_text=prompt,
                        api_call_id=resp.api_call_id, extra={"usage": resp.usage})
        out = parse_strict_json(resp.text)
        score = float(out.get("score", 0.0))
    except Exception as e:
        sample["reject_reason"] = f"veracity_judge_failed:{e}"
        sample["veracity"] = {"score": None, "n_independent_sources": independent_high,
                              "decision": "fail"}
        sample.setdefault("provenance", {})["stage_3a"] = prov.finalize()
        return

    decision = "pass" if score >= cfg["min_veracity_score"] else "fail"
    sample["veracity"] = {
        "score": score,
        "n_independent_sources": independent_high,
        "decision": decision,
        "rationale": out.get("rationale"),
        "time_sensitive": out.get("time_sensitive"),
        "valid_at": out.get("valid_at"),
    }
    if decision == "fail":
        sample["reject_reason"] = "veracity_low"
    sample.setdefault("provenance", {})["stage_3a"] = prov.finalize(score=score)

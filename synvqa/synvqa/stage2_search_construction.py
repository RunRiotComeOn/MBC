"""Stage 2 — Search-based Knowledge Construction.

For each anchor question:
  1. Rewrite into ≤ N search queries.
  2. Search + fetch top URLs, dedupe, rank by source reliability.
  3. Extract short-form answer + declarative fact via generator_llm.
  4. Apply rejection gates (answerable, not contested, ≥ min supporting snippets,
     atomic answer, citations present-verbatim in fetched text).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models.llm import LLMClient
from .models.search import SearchClient, fetch_url
from .utils.io import load_prompt, load_yaml, render_prompt
from .utils.json_parse import parse_strict_json
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_QR = _ROOT / "prompts" / "query_rewrite.txt"
_PROMPT_AE = _ROOT / "prompts" / "answer_extraction.txt"
_PROMPT_AT = _ROOT / "prompts" / "atomicity_judge.txt"


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _reliability_tier(domain: str, whitelist: dict) -> tuple[str, float]:
    if domain in set(whitelist.get("denylist", [])):
        return "deny", 0.0
    if domain in set(whitelist.get("high", [])):
        return "high", 1.0
    if domain in set(whitelist.get("medium", [])):
        return "medium", 0.6
    return "low", 0.3


def run(
    *,
    anchors: list[dict],
    output_dir: str | Path,
    config: dict[str, Any],
    generator_llm: LLMClient,
    judge_llm: LLMClient,
    search_client: SearchClient,
) -> list[dict]:
    log = get_logger("synvqa.stage2")
    sc = config["stage2"]
    reliability = load_yaml(_ROOT / sc["reliability_whitelist_path"]) if "reliability_whitelist_path" in sc else {}
    if not reliability:
        reliability = load_yaml(_ROOT / "configs" / "source_reliability.yaml")

    qr_tmpl = load_prompt(_PROMPT_QR)
    ae_tmpl = load_prompt(_PROMPT_AE)
    at_tmpl = load_prompt(_PROMPT_AT)

    out: list[dict] = []
    for anchor in anchors:
        sample = _process_one(
            anchor=anchor,
            config=sc,
            reliability=reliability,
            generator=generator_llm,
            judge=judge_llm,
            search=search_client,
            qr_tmpl=qr_tmpl,
            ae_tmpl=ae_tmpl,
            at_tmpl=at_tmpl,
            log=log,
        )
        out.append(sample)

    kept = [s for s in out if s.get("reject_reason") is None]
    log.info("stage 2: %d/%d kept", len(kept), len(out))
    return out


def _process_one(
    *,
    anchor: dict,
    config: dict,
    reliability: dict,
    generator: LLMClient,
    judge: LLMClient,
    search: SearchClient,
    qr_tmpl: str,
    ae_tmpl: str,
    at_tmpl: str,
    log,
) -> dict:
    prov = ProvenanceBuilder(stage=2)
    sample: dict[str, Any] = {
        "id": anchor["id"],
        "stage": 2,
        "anchor_question": anchor["question"],
        "retrieved_evidence": [],
        "answer": None,
        "fact": None,
        "reject_reason": None,
        "provenance": dict(anchor.get("provenance", {})),
    }

    # 1. Query rewrite
    prompt = render_prompt(qr_tmpl, question=anchor["question"], n=config["query_variants"])
    try:
        resp = generator.complete(prompt, temperature=0.2)
        prov.model_call(role="generator", model=resp.model, prompt_text=prompt,
                        api_call_id=resp.api_call_id, extra={"usage": resp.usage})
        queries = parse_strict_json(resp.text).get("queries", [])
        queries = [q for q in queries if isinstance(q, str) and q.strip()][: config["query_variants"]]
    except Exception as e:
        prov.log("query_rewrite_fallback", error=str(e))
        queries = [anchor["question"]]
    if not queries:
        queries = [anchor["question"]]

    # 2. Search + fetch
    hits: dict[str, dict] = {}   # url → hit meta
    for q in queries:
        try:
            for h in search.search(q):
                if h.url not in hits:
                    hits[h.url] = {"url": h.url, "title": h.title, "snippet": h.snippet,
                                   "query": q, "rank": h.rank,
                                   "raw_content": h.raw_content}
        except Exception as e:
            prov.log("search_error", query=q, error=str(e))

    if not hits:
        sample["reject_reason"] = "no_search_results"
        sample["provenance"]["stage_2"] = prov.finalize()
        return sample

    ranked = []
    for h in hits.values():
        dom = _domain(h["url"])
        tier, weight = _reliability_tier(dom, reliability)
        if tier == "deny":
            continue
        h2 = dict(h, domain=dom, reliability_tier=tier, reliability_weight=weight)
        ranked.append(h2)
    ranked.sort(key=lambda x: (-x["reliability_weight"], x["rank"]))
    ranked = ranked[: config["top_k_urls"]]

    # 3. Use Tavily raw_content directly (server-side fetch); fall back to
    #    local fetch_url only when raw_content is missing.
    max_chars = config.get("max_content_chars", 40000)
    evidence = []
    for h in ranked:
        rc = (h.get("raw_content") or "").strip()
        if rc:
            content = rc[:max_chars]
            prov.log("fetch", url=h["url"], ok=True, err=None, source="tavily_raw")
            evidence.append({**h, "content": content})
        else:
            fetched = fetch_url(
                h["url"],
                timeout_s=config["fetch_timeout_s"],
                max_chars=max_chars,
            )
            prov.log("fetch", url=h["url"], ok=fetched["ok"], err=fetched.get("error"),
                     source="local_fallback")
            if fetched["ok"]:
                evidence.append({**h, "content": fetched["text"]})
    if not evidence:
        sample["reject_reason"] = "all_fetches_failed"
        sample["provenance"]["stage_2"] = prov.finalize()
        return sample

    # 4. Answer extraction
    evidence_blocks = "\n\n".join(
        f"[{i}] url: {e['url']}\ncontent: {e['content']}"
        for i, e in enumerate(evidence)
    )
    prompt = render_prompt(ae_tmpl, question=anchor["question"],
                           evidence_blocks=evidence_blocks)
    try:
        resp = generator.complete(prompt, temperature=0.0, max_tokens=1024,
                                   response_format_json=True)
        prov.model_call(role="generator", model=resp.model, prompt_text=prompt,
                        api_call_id=resp.api_call_id, extra={"usage": resp.usage})
        extracted = parse_strict_json(resp.text)
    except Exception as e:
        prov.log("extraction_fallback", error=str(e))
        fallback_answer = getattr(search, "last_answer", "") or (
            evidence[0]["snippet"].split(".")[0].strip() if evidence and evidence[0].get("snippet") else ""
        )
        if not fallback_answer and evidence:
            fallback_answer = evidence[0]["title"] or evidence[0]["snippet"] or anchor["question"]
        extracted = {
            "answerable": bool(fallback_answer),
            "contested": False,
            "answer": fallback_answer,
            "fact_statement": fallback_answer or anchor["question"],
            "supporting_snippets": [
                {
                    "url": evidence[0]["url"],
                    "quote": (evidence[0]["snippet"] or fallback_answer)[:240],
                }
            ] if evidence else [],
            "confidence": 0.5,
            "notes": "local_fallback",
        }

    if not extracted.get("answerable"):
        sample["reject_reason"] = "not_answerable"
        sample["provenance"]["stage_2"] = prov.finalize(extracted=extracted)
        return sample
    if config["drop_contested"] and extracted.get("contested"):
        sample["reject_reason"] = "contested"
        sample["provenance"]["stage_2"] = prov.finalize(extracted=extracted)
        return sample

    supporting = extracted.get("supporting_snippets", []) or []
    if len(supporting) < config["min_supporting_snippets"]:
        sample["reject_reason"] = "insufficient_snippets"
        sample["provenance"]["stage_2"] = prov.finalize(extracted=extracted)
        return sample

    # Citation verification: quote must appear verbatim in fetched text.
    if config.get("citation_verify", True):
        url_to_text = {e["url"]: e["content"] for e in evidence}
        verified = []
        for s in supporting:
            url, quote = s.get("url"), s.get("quote", "")
            if url in url_to_text and quote and quote in url_to_text[url]:
                verified.append(s)
        if len(verified) < config["min_supporting_snippets"]:
            sample["reject_reason"] = "citation_verification_failed"
            sample["provenance"]["stage_2"] = prov.finalize(
                extracted=extracted,
                verified_count=len(verified),
            )
            return sample
        supporting = verified

    # Atomicity check
    at_prompt = render_prompt(at_tmpl, question=anchor["question"],
                              answer=extracted["answer"])
    try:
        at_resp = judge.complete(at_prompt, temperature=0.0,
                                 response_format_json=True)
        prov.model_call(role="judge", model=at_resp.model, prompt_text=at_prompt,
                        api_call_id=at_resp.api_call_id, extra={"usage": at_resp.usage})
        at_out = parse_strict_json(at_resp.text)
    except Exception as e:
        prov.log("atomicity_fallback", error=str(e))
        at_out = {
            "atomic": bool(extracted.get("answer")) and len(str(extracted.get("answer")).split()) <= 12,
            "reason": "local_fallback",
        }
    if not at_out.get("atomic"):
        sample["reject_reason"] = "answer_not_atomic"
        sample["provenance"]["stage_2"] = prov.finalize(
            extracted=extracted, atomicity=at_out,
        )
        return sample

    sample["retrieved_evidence"] = [
        {k: e[k] for k in ("url", "domain", "reliability_tier", "title", "snippet", "query")}
        for e in evidence
    ]
    sample["answer"] = extracted["answer"]
    sample["fact"] = extracted["fact_statement"]
    sample["supporting_snippets"] = supporting
    sample["extracted"] = {
        "confidence": extracted.get("confidence"),
        "contested": extracted.get("contested", False),
        "notes": extracted.get("notes"),
    }
    sample["provenance"]["stage_2"] = prov.finalize(n_evidence=len(evidence))
    return sample

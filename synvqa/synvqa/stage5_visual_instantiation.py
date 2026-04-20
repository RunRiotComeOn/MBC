"""Stage 5 — Visual Instantiation.

Renders the caption into an image and verifies caption↔image faithfulness:
  (a) captioning round-trip similarity ≥ τ,
  (b) targeted visual-anchor probe says "yes",
  (c) OCR-based answer-leakage check (T2I often renders text literally).

Final surviving samples are appended to output_dir/d_train.jsonl.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models.embeddings import EmbeddingClient
from .models.ocr import ocr_image
from .models.t2i import T2IClient
from .models.vlm import VLMClient
from .utils.io import append_jsonl, load_prompt, render_prompt
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT_FAITH = _ROOT / "prompts" / "faithfulness_vqa_probe.txt"


def run(
    *,
    samples: list[dict],
    output_dir: str | Path,
    config: dict[str, Any],
    t2i: T2IClient,
    faithfulness_vlm: VLMClient,
    embedder: EmbeddingClient,
) -> list[dict]:
    log = get_logger("synvqa.stage5")
    cfg = config["stage5"]
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    d_train = out_dir / "d_train.jsonl"

    faith_tmpl = load_prompt(_PROMPT_FAITH)

    for sample in samples:
        if sample.get("reject_reason"):
            continue
        if not sample.get("caption"):
            sample["reject_reason"] = "missing_caption"
            continue
        _instantiate(sample, cfg, t2i, faithfulness_vlm, embedder,
                     images_dir, faith_tmpl, d_train)

    kept = sum(1 for s in samples if s.get("reject_reason") is None
               and s.get("image_path"))
    log.info("stage 5: %d/%d kept", kept, len(samples))
    return samples


def _instantiate(
    sample: dict,
    cfg: dict,
    t2i: T2IClient,
    faith: VLMClient,
    embedder: EmbeddingClient,
    images_dir: Path,
    faith_tmpl: str,
    d_train: Path,
) -> None:
    prov = ProvenanceBuilder(stage=5)
    img_path = images_dir / f"{sample['id']}.png"
    caption = sample["caption"]
    vqa_answer = sample["vqa_answer"]
    anchor = sample["visual_anchor"]

    skip_faith = cfg.get("skip_faithfulness", False)

    for attempt in range(cfg["max_retries"] + 1):
        seed = 1000 + attempt * 17 + hash(sample["id"]) % 10000
        try:
            info = t2i.generate(caption, img_path, seed=seed)
            prov.log("t2i", attempt=attempt, seed=seed, path=info["path"])
        except Exception as e:
            prov.log("t2i_error", attempt=attempt, error=str(e))
            continue

        sim = 1.0
        if not skip_faith:
            # (1) Captioning round-trip similarity
            try:
                rt_caption = faith.caption_image(str(img_path))
                sim = embedder.similarity(caption, rt_caption)
            except Exception as e:
                prov.log("rt_caption_error", attempt=attempt, error=str(e))
                sim = 0.0
                rt_caption = ""
            prov.log("round_trip", sim=sim, rt_caption=rt_caption[:200])

            if sim < cfg["fidelity_threshold"]:
                continue

            # (2) Targeted visual-anchor probe
            probe_prompt = render_prompt(faith_tmpl, visual_anchor=anchor)
            try:
                pr = faith.generate_with_image(str(img_path), probe_prompt, max_new_tokens=8)
                yes = "yes" in pr.text.strip().lower()[:5]
            except Exception as e:
                prov.log("anchor_probe_error", attempt=attempt, error=str(e))
                yes = False
            prov.log("anchor_probe", yes=yes, text=(pr.text if yes else ""))
            if not yes:
                continue
        else:
            prov.log("skip_faithfulness", reason="config")

        # (3) OCR answer-in-image leakage
        if cfg.get("ocr_leakage_check", True):
            try:
                ocr_text = ocr_image(img_path)
            except Exception as e:
                prov.log("ocr_error", error=str(e))
                ocr_text = ""
            if vqa_answer.strip().lower() in ocr_text.lower():
                prov.log("ocr_leakage", attempt=attempt,
                         ocr_excerpt=ocr_text[:200])
                continue

        # All checks passed.
        sample["image_path"] = str(img_path)
        sample["faithfulness_score"] = float(sim)
        sample.setdefault("provenance", {})["stage_5"] = prov.finalize(
            attempts=attempt + 1, sim=sim,
        )
        append_jsonl(d_train, _clean_for_output(sample))
        return

    sample["reject_reason"] = "t2i_failed"
    sample.setdefault("provenance", {})["stage_5"] = prov.finalize(attempts=cfg["max_retries"] + 1)


def _clean_for_output(sample: dict) -> dict:
    """Drop large/intermediate fields to keep d_train.jsonl compact."""
    drop = {"retrieved_evidence", "supporting_snippets", "extracted"}
    return {k: v for k, v in sample.items() if k not in drop}

"""Stage 1 — Knowledge Anchor Pool.

Bootstraps a diverse pool of fact-seeking anchor questions via LLM expansion
from a human-authored seed set. Diversity enforced through embedding-based
near-duplicate filtering and a post-hoc k-means cluster audit.
"""
from __future__ import annotations

import datetime
import json
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .models.embeddings import EmbeddingClient
from .models.llm import LLMClient
from .utils.dedup import EmbeddingDeduper
from .utils.io import append_jsonl, load_prompt, read_jsonl, render_prompt, write_jsonl
from .utils.json_parse import parse_strict_json
from .utils.logging import get_logger
from .utils.provenance import ProvenanceBuilder

_ROOT = Path(__file__).resolve().parents[1]
_PROMPT = _ROOT / "prompts" / "anchor_expansion.txt"


def run(
    *,
    seeds_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    generator_llm: LLMClient,
    embedder: EmbeddingClient,
    rng: random.Random | None = None,
) -> list[dict]:
    log = get_logger("synvqa.stage1")
    rng = rng or random.Random(0)
    stage_cfg = config["stage1"]

    target_size = int(config["target_dataset_size"]) * int(config["oversample_factor"])
    n_seeds_min = int(stage_cfg["n_seeds_min"])
    k = int(stage_cfg["demos_per_call"])
    m = int(stage_cfg["new_per_call"])
    dedup_threshold = float(stage_cfg["dedup_threshold"])
    max_iters = int(stage_cfg["max_iters"])
    cluster_k = int(stage_cfg["cluster_k"])
    seed_mix_min = int(stage_cfg["seed_mix_min"])
    early_stop_n = int(stage_cfg["early_stop_consecutive_high_reject"])
    high_reject = float(stage_cfg["high_reject_threshold"])

    seeds = list(read_jsonl(seeds_path))
    if len(seeds) < n_seeds_min:
        raise ValueError(
            f"need ≥ {n_seeds_min} seeds but got {len(seeds)}: extend {seeds_path}"
        )
    seed_ids = {s["id"] for s in seeds}
    log.info("loaded %d seeds", len(seeds))

    prompt_template = load_prompt(_PROMPT)

    deduper = EmbeddingDeduper(embedder=embedder, threshold=dedup_threshold)
    deduper.seed(s["question"] for s in seeds)

    pool: list[dict] = [
        {"id": s["id"], "question": s["question"], "seed_ancestry": [s["id"]]}
        for s in seeds
    ]
    consecutive_high_reject = 0
    iteration = 0

    while len(pool) < target_size and iteration < max_iters:
        iteration += 1

        seed_demos = [p for p in pool if p["id"] in seed_ids]
        rng.shuffle(seed_demos)
        must_seed = seed_demos[: max(seed_mix_min, 2)]
        remaining = [p for p in pool if p["id"] not in {d["id"] for d in must_seed}]
        rng.shuffle(remaining)
        demos = must_seed + remaining[: k - len(must_seed)]
        demos_text = "\n".join(f"- {d['question']}" for d in demos)

        prompt = render_prompt(
            prompt_template,
            k=k, m=m, demos=demos_text,
            cutoff_date=config.get("cutoff_date", "2025-02"),
            now_date=datetime.date.today().isoformat(),
        )
        prov = ProvenanceBuilder(stage=1)
        try:
            resp = generator_llm.complete(prompt, temperature=0.8)
            prov.model_call(role="generator", model=resp.model, prompt_text=prompt,
                            api_call_id=resp.api_call_id,
                            extra={"usage": resp.usage})
            payload = parse_strict_json(resp.text)
            candidates = [q["question"].strip() for q in payload.get("questions", [])
                          if "question" in q and q["question"].strip()]
        except Exception as e:
            log.warning("iter %d: generation failed: %s", iteration, e)
            continue

        pre = len(candidates)
        kept, rejected = deduper.filter_and_add(candidates)
        reject_rate = (len(rejected) / pre) if pre else 1.0
        log.info(
            "iter %d: demos=%d cand=%d kept=%d rej=%d reject_rate=%.2f pool=%d/%d",
            iteration, len(demos), pre, len(kept), len(rejected), reject_rate,
            len(pool), target_size,
        )

        demo_ids = [d["id"] for d in demos]
        for q in kept:
            pool.append({
                "id": f"anchor_{uuid.uuid4().hex[:10]}",
                "question": q,
                "seed_ancestry": demo_ids,
                "provenance": {"stage_1": prov.finalize(iteration=iteration)},
            })

        if reject_rate > high_reject:
            consecutive_high_reject += 1
        else:
            consecutive_high_reject = 0
        if consecutive_high_reject >= early_stop_n:
            log.warning("dedup collapse: %d consecutive rounds with reject_rate>%.2f — stopping",
                        consecutive_high_reject, high_reject)
            break

    pool = pool[:target_size]

    out_dir = Path(output_dir)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    diversity = _cluster_audit(deduper.vectors[: len(pool)], cluster_k)
    (out_dir / "metadata" / "anchor_diversity.json").write_text(
        json.dumps(diversity, indent=2)
    )
    if diversity["n_nonempty_clusters"] < 15:
        log.warning("seed too narrow: only %d non-empty clusters (want ≥ 15). "
                    "Consider expanding seeds.", diversity["n_nonempty_clusters"])

    log.info("stage 1 done: %d anchors", len(pool))
    return pool


def _cluster_audit(X: np.ndarray, k: int) -> dict[str, Any]:
    if X.shape[0] < k:
        k = max(2, X.shape[0] // 2)
    try:
        from sklearn.cluster import KMeans  # type: ignore
        km = KMeans(n_clusters=k, n_init=4, random_state=0)
        labels = km.fit_predict(X)
    except Exception:
        # Fallback: no clustering if sklearn unavailable.
        labels = np.zeros(X.shape[0], dtype=int)
    counts: dict[int, int] = {}
    for l in labels.tolist():
        counts[l] = counts.get(l, 0) + 1
    return {
        "k": k,
        "n_nonempty_clusters": sum(1 for v in counts.values() if v > 0),
        "cluster_sizes": sorted(counts.values(), reverse=True),
    }

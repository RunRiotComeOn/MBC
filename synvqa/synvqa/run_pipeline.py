"""Pipeline orchestrator.

Usage:
    python -m synvqa.run_pipeline \
        --config configs/pipeline.yaml \
        --seeds seeds/anchor_seeds.jsonl \
        --output data/synthetic_vqa \
        --stages 1,2,3,4,5 \
        --resume-from-checkpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models.embeddings import EmbeddingClient
from .models.llm import LLMClient
from .models.search import SearchClient
from .models.t2i import T2IClient
from .models.vlm import VLMClient
from .utils.checkpoint import CheckpointManager
from .utils.io import load_env_file, load_yaml, read_jsonl, write_jsonl
from .utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="synvqa pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stages", default="1,2,3,4,5",
                   help="Comma-separated stage ids (1,2,3a,3b,4,5 — '3' = 3a,3b)")
    p.add_argument("--resume-from-checkpoint", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Load configs & clients but exit before running stages.")
    return p.parse_args()


def _parse_stages(s: str) -> list[str]:
    out: list[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "3":
            out.extend(["3a", "3b"])
        elif tok in {"1", "2", "3a", "3b", "4", "5"}:
            out.append(tok)
        else:
            raise ValueError(f"unknown stage: {tok}")
    return out


def _assert_decorrelation(config: dict) -> None:
    if not config.get("strict_model_decorrelation", True):
        return
    # The spec requires different model families for generator / judge / target / faith.
    def family(m: str) -> str:
        m = m.lower()
        if m.startswith("claude"):
            return "anthropic"
        if m.startswith("deepseek"):
            return "deepseek"
        if m.startswith(("gpt", "o1", "o3", "o4")):
            return "openai"
        if m.startswith(("qwen", "internvl", "llava", "llama", "mistral", "flux")):
            return "oss"
        return "other"
    fams = {
        "generator": family(config["generator_llm"]),
        "judge": family(config["judge_llm"]),
        "target": family(config["target_vlm"]),
        "faith": family(config["faithfulness_vlm"]),
    }
    if fams["generator"] == fams["judge"]:
        raise ValueError(f"generator and judge must differ in family (got {fams})")
    # target/faith may share family (both OSS) but should not equal generator/judge.


def main() -> int:
    args = parse_args()
    load_env_file()
    config = load_yaml(args.config)
    # resolve output_dir from CLI, overriding config if provided
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger("synvqa",
                     level=config.get("logging", {}).get("level", "INFO"),
                     file=str(output_dir / "pipeline.log"))

    _assert_decorrelation(config)
    stages = _parse_stages(args.stages)
    log.info("running stages: %s", stages)

    ck = CheckpointManager(output_dir)

    # Build clients lazily — only construct what we need.
    need_generator = any(s in stages for s in ("1", "2", "3b", "4"))
    need_judge = any(s in stages for s in ("2", "3a", "3b", "4"))
    need_embedder = any(s in stages for s in ("1", "4", "5"))
    need_search = "2" in stages
    need_target = "3b" in stages
    need_t2i = "5" in stages
    need_faith = "5" in stages

    generator_llm = LLMClient(config["generator_llm"]) if need_generator else None
    judge_llm = LLMClient(config["judge_llm"]) if need_judge else None
    embedder = EmbeddingClient(config.get("embedding_model", "text-embedding-3-large")) if need_embedder else None
    search_client = SearchClient(
        backend=config["stage2"]["search_backend"],
        results_per_query=config["stage2"]["results_per_query"],
    ) if need_search else None
    target_vlm = VLMClient(config["target_vlm"]) if need_target else None
    t2i = T2IClient(
        model=config["t2i_model"],
        resolution=config["stage5"]["t2i_resolution"],
        steps=config["stage5"]["t2i_steps"],
    ) if need_t2i else None
    faithfulness_vlm = VLMClient(config["faithfulness_vlm"]) if need_faith else None

    if args.dry_run:
        log.info("dry run — exiting before stage execution")
        return 0

    # -------- Stage 1 --------
    samples: list[dict] | None = None
    if "1" in stages:
        cached = ck.load_if_resuming(1, args.resume_from_checkpoint)
        if cached is not None:
            log.info("stage 1 resumed from checkpoint: %d items", len(cached))
            anchors = cached
        else:
            from .stage1_anchor_pool import run as run_s1
            anchors = run_s1(
                seeds_path=args.seeds, output_dir=output_dir, config=config,
                generator_llm=generator_llm, embedder=embedder,
            )
            ck.write(1, anchors)
        samples = anchors
    else:
        if ck.exists(1):
            samples = list(ck.read(1))
            log.info("loaded stage 1 checkpoint: %d items", len(samples))

    # -------- Stage 2 --------
    if "2" in stages:
        cached = ck.load_if_resuming(2, args.resume_from_checkpoint)
        if cached is not None:
            log.info("stage 2 resumed: %d items", len(cached))
            samples = cached
        else:
            assert samples is not None, "stage 2 requires stage 1 output"
            from .stage2_search_construction import run as run_s2
            samples = run_s2(
                anchors=samples, output_dir=output_dir, config=config,
                generator_llm=generator_llm, judge_llm=judge_llm,
                search_client=search_client,
            )
            ck.write(2, samples)
    elif samples is None and ck.exists(2):
        samples = list(ck.read(2))

    # -------- Stage 3a --------
    if "3a" in stages:
        cached = ck.load_if_resuming(31, args.resume_from_checkpoint)
        if cached is not None:
            samples = cached
        else:
            assert samples is not None, "stage 3a requires stage 2 output"
            from .stage3a_veracity import run as run_s3a
            samples = run_s3a(samples=samples, config=config, judge_llm=judge_llm)
            ck.write(31, samples)
    elif samples is None and ck.exists(31):
        samples = list(ck.read(31))

    # -------- Stage 3b --------
    if "3b" in stages:
        cached = ck.load_if_resuming(32, args.resume_from_checkpoint)
        if cached is not None:
            samples = cached
        else:
            assert samples is not None, "stage 3b requires stage 3a output"
            from .stage3b_probe import run as run_s3b
            samples = run_s3b(
                samples=samples, config=config,
                target_vlm=target_vlm, judge_llm=judge_llm,
                generator_llm=generator_llm,
            )
            ck.write(32, samples)
    elif samples is None and ck.exists(32):
        samples = list(ck.read(32))

    # -------- Stage 4 --------
    if "4" in stages:
        cached = ck.load_if_resuming(4, args.resume_from_checkpoint)
        if cached is not None:
            samples = cached
        else:
            assert samples is not None, "stage 4 requires stage 3b output"
            from .stage4_vqa_synthesis import run as run_s4
            samples = run_s4(
                samples=samples, output_dir=output_dir, config=config,
                generator_llm=generator_llm, judge_llm=judge_llm,
                embedder=embedder,
            )
            ck.write(4, samples)
    elif samples is None and ck.exists(4):
        samples = list(ck.read(4))

    # -------- Stage 5 --------
    if "5" in stages:
        cached = ck.load_if_resuming(5, args.resume_from_checkpoint)
        if cached is not None:
            samples = cached
        else:
            assert samples is not None, "stage 5 requires stage 4 output"
            from .stage5_visual_instantiation import run as run_s5
            samples = run_s5(
                samples=samples, output_dir=output_dir, config=config,
                t2i=t2i, faithfulness_vlm=faithfulness_vlm,
                embedder=embedder,
            )
            ck.write(5, samples)

    # Summary
    if samples is not None:
        kept = sum(1 for s in samples if s.get("reject_reason") is None
                   and s.get("image_path"))
        log.info("pipeline complete — %d final samples (of %d processed)",
                 kept, len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())

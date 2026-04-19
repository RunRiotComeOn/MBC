# synvqa — Knowledge-Grounded Synthetic VQA Generation

Pipeline for generating a large-scale synthetic VQA dataset grounded in
external factual statements that lie outside the target VLM's parametric
knowledge. Follows the 5-stage design:

1. **Knowledge Anchor Pool** — bootstrap diverse fact-seeking questions from seeds.
2. **Search-based Knowledge Construction** — retrieve + extract declarative facts.
3. **Validation** — (3a) veracity judge, (3b) parametric-gap probe.
4. **VQA Synthesis** — image-grounded caption + Q/A triplet with 3-constraint verification.
5. **Visual Instantiation** — T2I rendering + caption/image faithfulness checks.

## Layout

```
configs/        pipeline.yaml, source_reliability.yaml
prompts/        11 prompt templates (anchor_expansion, vqa_synthesis, ...)
seeds/          anchor_seeds.jsonl
synvqa/         package code
  models/       LLM, VLM, T2I, search, embeddings, OCR clients
  utils/        io, logging, dedup, provenance, checkpoint, json_parse
  stage{1..5}*.py
  run_pipeline.py
data/           generated images + d_train.jsonl
```

## Run

```bash
python -m synvqa.run_pipeline \
    --config configs/pipeline.yaml \
    --seeds seeds/anchor_seeds.jsonl \
    --output data/synthetic_vqa \
    --stages 1,2,3,4,5 \
    --resume-from-checkpoint
```

Environment variables:
- `ANTHROPIC_API_KEY` — for the generator LLM (Claude).
- `OPENAI_API_KEY` — for the judge LLM and embeddings.
- `DEEPSEEK_API_KEY` — for DeepSeek models through the OpenAI-compatible SDK.
- `DEEPSEEK_BASE_URL` — optional, defaults to `https://api.deepseek.com`.
- `TAVILY_API_KEY` / `SERPAPI_API_KEY` / `BING_API_KEY` — for the chosen search backend.

You can also place these values in a local `.env` file at the repository root;
the pipeline loads it automatically if present.

The target VLM (`Qwen2.5-VL-7B-Instruct`), faithfulness VLM (`InternVL2-8B`),
and T2I model (`FLUX.1-dev`) are loaded via HuggingFace Transformers /
Diffusers. For production, point `VLMClient` / `T2IClient` at your deployed
inference endpoint.

## Model decorrelation rule

Enforced at startup: `generator_llm`, `judge_llm`, `target_vlm`,
`faithfulness_vlm` must all come from distinct model families to avoid
correlated errors.

## Checkpoints

Each stage writes `data/synthetic_vqa/stage_{i}_checkpoint.jsonl`. Resume
mid-pipeline with `--resume-from-checkpoint`. Stage 3a checkpoint is `stage_31_*`,
Stage 3b is `stage_32_*`.

## Expected yield

Empirical rule-of-thumb end-to-end: ~14%. `oversample_factor: 8` in
`configs/pipeline.yaml` compensates.

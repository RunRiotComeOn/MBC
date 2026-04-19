"""Text-to-image client.

Default target: FLUX.1-dev via diffusers. Replace with a served endpoint if
you run a dedicated T2I service.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class T2IClient:
    def __init__(self, model: str = "FLUX.1-dev", device: str | None = None,
                 resolution: int = 1024, steps: int = 30):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.steps = steps
        self._pipe = None

    def _lazy_load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from diffusers import FluxPipeline  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "diffusers / torch not installed; install or point T2IClient at a served endpoint"
            ) from e
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._pipe = FluxPipeline.from_pretrained(self.model, torch_dtype=dtype)
        if torch.cuda.is_available():
            self._pipe = self._pipe.to("cuda")

    def generate(
        self,
        caption: str,
        out_path: str | Path,
        *,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> dict[str, Any]:
        self._lazy_load()
        import torch  # type: ignore

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._pipe.device).manual_seed(seed)
        img = self._pipe(
            prompt=caption,
            num_inference_steps=self.steps,
            height=self.resolution,
            width=self.resolution,
            generator=gen,
        ).images[0]
        img.save(out_path)
        return {"path": str(out_path), "seed": seed, "model": self.model}

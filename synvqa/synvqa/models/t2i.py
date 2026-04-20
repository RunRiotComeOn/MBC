"""Text-to-image client.

Supports three backends:
  - "pollinations"  — free, no API key, calls image.pollinations.ai
  - "gemini"        — Gemini-compatible API (generateContent with IMAGE modality)
  - anything else   — local FLUX.1-dev via diffusers (needs GPU + torch)

For gemini backend, set env vars:
  GEMINI_T2I_BASE_URL — API base URL (e.g. https://grsai.dakka.com.cn)
  GEMINI_T2I_API_KEY  — API key
  GEMINI_T2I_MODEL    — model name (e.g. nano-banana-fast)
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..utils.io import load_env_file


class T2IClient:
    def __init__(self, model: str = "FLUX.1-dev", device: str | None = None,
                 resolution: int = 1024, steps: int = 30):
        self.model = model
        self.device = device
        self.resolution = resolution
        self.steps = steps
        self._pipe = None
        m = model.lower()
        if m == "pollinations":
            self._backend = "pollinations"
        elif m == "gemini":
            self._backend = "gemini"
        else:
            self._backend = "local"

    # ---- Gemini-compatible backend ----

    def _generate_gemini(
        self,
        caption: str,
        out_path: Path,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        import requests  # type: ignore

        load_env_file()
        base_url = os.environ.get("GEMINI_T2I_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("GEMINI_T2I_API_KEY", "")
        model_name = os.environ.get("GEMINI_T2I_MODEL", "nano-banana-fast")

        if not base_url or not api_key:
            raise RuntimeError("GEMINI_T2I_BASE_URL and GEMINI_T2I_API_KEY must be set")

        url = f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": caption}]}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        session = requests.Session()
        session.trust_env = False
        resp = session.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            raise RuntimeError(f"Gemini API error: {data['error']}")

        # Extract image from response
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                img_bytes = base64.b64decode(part["inlineData"]["data"])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(img_bytes)
                return {"path": str(out_path), "seed": seed, "model": model_name}

        raise RuntimeError("Gemini API returned no image data")

    # ---- Pollinations backend ----

    def _generate_pollinations(
        self,
        caption: str,
        out_path: Path,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        import subprocess

        params = f"width={self.resolution}&height={self.resolution}&nologo=true"
        if seed is not None:
            params += f"&seed={seed}"

        url = f"https://image.pollinations.ai/prompt/{quote(caption)}?{params}"
        result = subprocess.run(
            ["curl", "-s", "-f", "--noproxy", "*", "-o", str(out_path),
             "--max-time", "120", url],
            capture_output=True, text=True, timeout=130,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.strip()}")
        return {"path": str(out_path), "seed": seed, "model": "pollinations"}

    # ---- Local diffusers backend ----

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

    def _generate_local(
        self,
        caption: str,
        out_path: Path,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        self._lazy_load()
        import torch  # type: ignore

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

    # ---- Public API ----

    def generate(
        self,
        caption: str,
        out_path: str | Path,
        *,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> dict[str, Any]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._backend == "gemini":
            return self._generate_gemini(caption, out_path, seed=seed)
        if self._backend == "pollinations":
            return self._generate_pollinations(caption, out_path, seed=seed)
        return self._generate_local(caption, out_path, seed=seed)

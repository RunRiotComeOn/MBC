"""Vision-language model wrapper.

Two roles:
  - target_vlm (Stage 3b): probed for parametric knowledge. Text-only input,
    must expose logprobs where possible (HF transformers backend).
  - faithfulness_vlm (Stage 5): image + short question → short answer.

Supports two backends:
  - API models (deepseek-*, gpt-*, claude-*): routed through OpenAI-compatible
    SDK, same as LLMClient.  Vision requests encode the image as base64.
  - Local HF models: loaded via transformers (needs GPU + weights).
"""
from __future__ import annotations

import base64
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import load_env_file


@dataclass
class VLMResponse:
    text: str
    logprob_confidence: float | None = None
    samples: list[str] = field(default_factory=list)
    model: str = ""
    raw: Any = None


def _is_api_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("deepseek", "gpt", "o1", "o3", "o4", "claude"))


class VLMClient:
    def __init__(self, model: str, device: str | None = None):
        self.model = model
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self._hf = None
        self._proc = None
        self._api_client = None
        self._backend = "api" if _is_api_model(model) else "local"

    # ===================== API backend =====================

    def _ensure_api_client(self) -> None:
        if self._api_client is not None:
            return
        load_env_file()
        from openai import OpenAI  # type: ignore
        m = self.model.lower()
        if m.startswith("deepseek"):
            self._api_client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url=os.environ.get("DEEPSEEK_BASE_URL",
                                        "https://api.deepseek.com"),
            )
        elif m.startswith("claude"):
            import anthropic  # type: ignore
            self._api_client = anthropic.Anthropic()
        else:
            self._api_client = OpenAI()

    def _api_generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        return_logprobs: bool = False,
    ) -> VLMResponse:
        self._ensure_api_client()
        resp = self._api_client.chat.completions.create(
            model=self.model,
            temperature=temperature if temperature > 0.0 else 0.0,
            max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return VLMResponse(text=text.strip(), model=self.model, raw=resp)

    def _api_sample_texts(self, prompt: str, *, n: int,
                          temperature: float = 0.7,
                          max_new_tokens: int = 64) -> list[str]:
        outs = []
        for _ in range(n):
            r = self._api_generate_text(
                prompt, temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            outs.append(r.text)
        return outs

    def _api_generate_with_image(
        self,
        image_path: str,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> VLMResponse:
        self._ensure_api_client()
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg",
                "jpeg": "image/jpeg"}.get(ext, "image/png")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
        resp = self._api_client.chat.completions.create(
            model=self.model,
            max_tokens=max_new_tokens,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        return VLMResponse(text=text.strip(), model=self.model, raw=resp)

    def _api_caption_image(self, image_path: str) -> str:
        r = self._api_generate_with_image(
            image_path,
            "Write a one-paragraph detailed caption of this image.",
            max_new_tokens=256,
        )
        return r.text

    # ===================== Local HF backend =====================

    def _lazy_load(self) -> None:
        if self._hf is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed; install it or point VLMClient at a served endpoint"
            ) from e
        self._proc = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        self._hf = AutoModelForCausalLM.from_pretrained(
            self.model, trust_remote_code=True, torch_dtype="auto"
        ).to(self.device)
        self._hf.eval()

    def _local_generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        return_logprobs: bool = True,
    ) -> VLMResponse:
        self._lazy_load()
        import torch  # type: ignore

        inputs = self._proc(text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._hf.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
                return_dict_in_generate=True,
                output_scores=return_logprobs,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self._proc.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()

        confidence: float | None = None
        if return_logprobs and getattr(out, "scores", None):
            logp = 0.0
            n = 0
            for t_idx, score in enumerate(out.scores):
                tok = gen_ids[t_idx].item()
                probs = torch.log_softmax(score[0], dim=-1)
                logp += float(probs[tok].item())
                n += 1
            if n > 0:
                confidence = math.exp(logp / n)

        return VLMResponse(text=text, logprob_confidence=confidence,
                           model=self.model, raw=out)

    def _local_generate_with_image(
        self,
        image_path: str,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> VLMResponse:
        self._lazy_load()
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(image_path).convert("RGB")
        inputs = self._proc(text=prompt, images=img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._hf.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
        text = self._proc.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()
        return VLMResponse(text=text, model=self.model, raw=out)

    # ===================== Public API =====================

    def generate_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        return_logprobs: bool = True,
    ) -> VLMResponse:
        if self._backend == "api":
            return self._api_generate_text(
                prompt, temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_logprobs=return_logprobs,
            )
        return self._local_generate_text(
            prompt, temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_logprobs=return_logprobs,
        )

    def sample_texts(self, prompt: str, *, n: int, temperature: float = 0.7,
                     max_new_tokens: int = 64) -> list[str]:
        if self._backend == "api":
            return self._api_sample_texts(
                prompt, n=n, temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        outs = []
        for _ in range(n):
            r = self._local_generate_text(
                prompt, temperature=temperature,
                max_new_tokens=max_new_tokens, return_logprobs=False,
            )
            outs.append(r.text)
        return outs

    def generate_with_image(
        self,
        image_path: str,
        prompt: str,
        *,
        max_new_tokens: int = 64,
    ) -> VLMResponse:
        if self._backend == "api":
            return self._api_generate_with_image(
                image_path, prompt, max_new_tokens=max_new_tokens,
            )
        return self._local_generate_with_image(
            image_path, prompt, max_new_tokens=max_new_tokens,
        )

    def caption_image(self, image_path: str) -> str:
        if self._backend == "api":
            return self._api_caption_image(image_path)
        r = self._local_generate_with_image(
            image_path,
            "Write a one-paragraph detailed caption of this image.",
            max_new_tokens=256,
        )
        return r.text


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False

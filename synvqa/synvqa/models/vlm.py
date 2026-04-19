"""Vision-language model wrapper.

Two roles:
  - target_vlm (Stage 3b): probed for parametric knowledge. Text-only input,
    must expose logprobs where possible (HF transformers backend).
  - faithfulness_vlm (Stage 5): image + short question → short answer.

Implementation notes:
  The HF transformers path is a best-effort wrapper around Qwen2.5-VL and
  InternVL; concrete loading code is guarded so imports don't fail in env
  without the weights. Replace .generate_text() / .generate_with_image()
  with your deployed inference endpoint if you serve these models elsewhere.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VLMResponse:
    text: str
    logprob_confidence: float | None = None
    samples: list[str] = field(default_factory=list)
    model: str = ""
    raw: Any = None


class VLMClient:
    def __init__(self, model: str, device: str | None = None):
        self.model = model
        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self._hf = None
        self._proc = None
        self._lock = False  # simple reentrancy guard

    # ------------- loading -------------
    def _lazy_load(self) -> None:
        if self._hf is not None:
            return
        # Actual loading deferred to your infra. We raise a clear error
        # rather than silently stubbing, to avoid fabricated outputs.
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

    # ------------- Stage 3b -------------
    def generate_text(
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
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
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

        return VLMResponse(text=text, logprob_confidence=confidence, model=self.model, raw=out)

    def sample_texts(self, prompt: str, *, n: int, temperature: float = 0.7,
                     max_new_tokens: int = 64) -> list[str]:
        outs = []
        for _ in range(n):
            r = self.generate_text(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                return_logprobs=False,
            )
            outs.append(r.text)
        return outs

    # ------------- Stage 5 -------------
    def generate_with_image(
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
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1] :]
        text = self._proc.batch_decode([gen_ids], skip_special_tokens=True)[0].strip()
        return VLMResponse(text=text, model=self.model, raw=out)

    def caption_image(self, image_path: str) -> str:
        r = self.generate_with_image(
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

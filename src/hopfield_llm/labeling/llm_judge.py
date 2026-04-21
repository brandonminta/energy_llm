"""LLM-judge labeler for factual accuracy assessment.

LLMJudge supports two backends:
  a) OpenAI-compatible REST API  (api_base is not None)
  b) Local HuggingFace model     (api_base is None, fallback for air-gapped HPC)

All judge outputs are returned verbatim so samples can be re-labelled with a
different parser without re-querying the judge.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace
from pathlib import Path

from hopfield_llm.datasets.base import DataSample
from hopfield_llm.utils.logging import get_logger

log = get_logger("labeling.llm_judge")

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_TEMPLATE = "convcoa_v1.txt"


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------


def build_judge_prompt(
    question: str,
    generated_text: str,
    gold_answers: list[str],
    template_path: str | Path = _PROMPTS_DIR / _DEFAULT_TEMPLATE,
) -> str:
    """Fill the judge prompt template with question/answer data.

    Args:
        question:        The question posed to the model.
        generated_text:  The model's generated answer to evaluate.
        gold_answers:    Reference (ground-truth) answers; joined with " | ".
        template_path:   Path to the .txt prompt template.

    Returns:
        Filled prompt string ready to send to the judge.
    """
    template = Path(template_path).read_text(encoding="utf-8")
    ground_truth = " | ".join(gold_answers) if gold_answers else ""
    return template.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_text,
    )


def _template_hash(template_path: str | Path) -> str:
    content = Path(template_path).read_text(encoding="utf-8")
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ------------------------------------------------------------------
# Output parser
# ------------------------------------------------------------------


def _parse_judge_output(raw: str) -> int | None:
    """Extract the last '0' or '1' from the final non-empty line.

    Returns 0, 1, or None (parse failure).
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    final_line = lines[-1]
    for char in reversed(final_line):
        if char in ("0", "1"):
            return int(char)
    return None


# ------------------------------------------------------------------
# LLMJudge class
# ------------------------------------------------------------------


class LLMJudge:
    """External LLM judge for binary factual-accuracy labeling.

    Backends
    --------
    api_base is not None  → OpenAI-compatible REST API (openai SDK required).
    api_base is None      → Local HuggingFace model via transformers.

    All outputs are returned verbatim; parsing is handled by label_sample_with_judge.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_base: str | None = "https://api.openai.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        template: str = _DEFAULT_TEMPLATE,
    ):
        self.model_id    = model_id
        self.api_base    = api_base
        self.api_key     = api_key
        self.temperature = temperature
        self.max_retries = max_retries

        self._template_path = _PROMPTS_DIR / template
        self._prompt_hash   = _template_hash(self._template_path)
        self._use_openai    = api_base is not None

        # Lazily initialised HF model (local backend only)
        self._hf_model     = None
        self._hf_tokenizer = None

    # ── Public API ──────────────────────────────────────────────────────────

    def judge(
        self,
        question: str,
        generated_text: str,
        gold_answers: list[str],
    ) -> dict:
        """Query the judge and return a result dict.

        Returns:
            {"raw": str, "parsed": int | None, "success": bool}
            "parsed" is 0 or 1 when successfully parsed, else None.
        """
        prompt = build_judge_prompt(
            question, generated_text, gold_answers, self._template_path
        )
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                if self._use_openai:
                    raw = self._call_openai_api(prompt)
                else:
                    raw = self._call_local_hf(prompt)
                parsed = _parse_judge_output(raw)
                return {"raw": raw, "parsed": parsed, "success": True}
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Judge attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)   # exponential back-off

        err_msg = f"Judge failed after {self.max_retries} attempts: {last_exc}"
        log.error(err_msg)
        raise RuntimeError(err_msg)

    # ── Backends ────────────────────────────────────────────────────────────

    def _call_openai_api(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai SDK is required for the API backend. "
                "Install with: pip install openai"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=32,
        )
        return response.choices[0].message.content or ""

    def _call_local_hf(self, prompt: str) -> str:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the local HF backend. "
                "Install with: pip install transformers"
            ) from exc

        if self._hf_tokenizer is None:
            log.info("Loading local HF judge model: %s", self.model_id)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._hf_model     = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype="auto"
            )
            self._hf_model.eval()

        import torch
        inputs  = self._hf_tokenizer(prompt, return_tensors="pt")
        inputs  = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        n_input = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )
        gen_ids = output_ids[0, n_input:]
        return self._hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ------------------------------------------------------------------
# Per-sample labeler
# ------------------------------------------------------------------


def label_sample_with_judge(
    sample: DataSample,
    judge: LLMJudge,
) -> DataSample:
    """Return a new DataSample labelled by the LLM judge.

    Does NOT mutate the input sample.

    Parsing contract:
    - parsed == 1  → correctness_score=1.0, is_hallucination=False
    - parsed == 0  → correctness_score=0.0, is_hallucination=True
    - parse failure → is_hallucination=None, label_source includes "__parse_failed"
    - judge error   → is_hallucination=None, label_source includes "__error"

    is_hallucination is NEVER silently set to False on any failure path.
    """
    base_label = f"llm_judge_v1"

    try:
        result = judge.judge(
            question=sample.question,
            generated_text=sample.generated_text,
            gold_answers=sample.gold_answers,
        )
    except RuntimeError as exc:
        return replace(
            sample,
            judge_raw_output=str(exc),
            judge_model=judge.model_id,
            judge_prompt_hash=judge._prompt_hash,
            label_source=f"{base_label}__error",
            is_hallucination=None,
            correctness_score=None,
        )

    raw    = result["raw"]
    parsed = result["parsed"]

    if parsed is None:
        return replace(
            sample,
            judge_raw_output=raw,
            judge_model=judge.model_id,
            judge_prompt_hash=judge._prompt_hash,
            label_source=f"{base_label}__parse_failed",
            is_hallucination=None,
            correctness_score=None,
        )

    return replace(
        sample,
        judge_raw_output=raw,
        judge_model=judge.model_id,
        judge_prompt_hash=judge._prompt_hash,
        label_source=base_label,
        label_version=judge._prompt_hash,
        correctness_score=float(parsed),
        is_hallucination=(parsed == 0),
    )

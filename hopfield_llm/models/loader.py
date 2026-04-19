"""Hugging Face causal LLM loader with 4-bit quantization support."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hopfield_llm.models.arch import resolve_backbone, resolve_layers
from hopfield_llm.models.profiles import (
    DEFAULT_MODEL_ALIAS,
    ModelProfile,
    resolve_model_profile,
)
from hopfield_llm.utils.logging import get_logger

log = get_logger("models.loader")


class HFLLM:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ALIAS,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        trust_remote_code: bool = True,
        output_hidden_states: bool = True,
    ):
        self.alias = model_id
        self.profile: Optional[ModelProfile] = resolve_model_profile(model_id)
        self.model_id: str = self.profile.model_id if self.profile is not None else model_id

        self.device: str = self._resolve_device(device)
        self.dtype: torch.dtype = self._resolve_dtype(dtype, self.device)
        self.trust_remote_code = trust_remote_code
        self.output_hidden_states = output_hidden_states

        self.load_in_4bit_requested = load_in_4bit
        self.load_in_4bit_effective = self._resolve_effective_4bit(
            requested=load_in_4bit,
            device=self.device,
            profile=self.profile,
        )

        self.tokenizer = self._load_tokenizer()
        self.quant_config = self._build_quant_config()
        self.model = self._load_model()
        self.model.eval()
        self.backbone = self._resolve_backbone()
        self.layers = self._resolve_layers()
        self.hidden_size = getattr(self.model.config, "hidden_size", None)
        self.num_hidden_layers = getattr(
            self.model.config, "num_hidden_layers", len(self.layers)
        )
        self._log_load_summary()

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(dtype: Optional[torch.dtype], device: str) -> torch.dtype:
        if dtype is not None:
            return dtype
        return torch.float16 if device == "cuda" else torch.float32

    @staticmethod
    def _resolve_effective_4bit(
        requested: bool, device: str, profile: Optional[ModelProfile]
    ) -> bool:
        if not requested:
            return False
        if device != "cuda":
            log.warning(
                "4-bit quantization requested but device=%s — falling back to full precision.",
                device,
            )
            return False
        if profile is not None and not profile.safe_4bit:
            log.warning(
                "4-bit quantization disabled: safe_4bit=False for profile '%s'.",
                profile.model_id,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def _build_quant_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.load_in_4bit_effective:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def _maybe_resize_embeddings(self, model):
        input_emb = model.get_input_embeddings()
        n_embed = input_emb.num_embeddings
        n_tok = len(self.tokenizer)
        needs_resize = n_tok > n_embed
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None and pad_id >= n_embed:
            needs_resize = True
        if needs_resize:
            model.resize_token_embeddings(n_tok)

    def _load_model(self):
        model_kwargs = dict(
            pretrained_model_name_or_path=self.model_id,
            output_hidden_states=self.output_hidden_states,
            trust_remote_code=self.trust_remote_code,
        )
        if self.quant_config is not None:
            model_kwargs["quantization_config"] = self.quant_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self.dtype
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        if self.quant_config is None and self.device != "cuda":
            model.to(self.device)
        self._maybe_resize_embeddings(model)
        return model

    # ------------------------------------------------------------------
    # Architecture resolution
    # ------------------------------------------------------------------

    def _resolve_backbone(self):
        return resolve_backbone(self.model)

    def _resolve_layers(self):
        return resolve_layers(self.backbone)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def forward(self, text: str):
        inputs = self.tokenize(text)
        return self.model(**inputs)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **generate_kwargs,
    ) -> str:
        inputs = self.tokenize(prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "output_hidden_states": self.output_hidden_states,
            "load_in_4bit_requested": self.load_in_4bit_requested,
            "load_in_4bit_effective": self.load_in_4bit_effective,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "n_layers_resolved": self.n_layers,
            "profile": (
                None
                if self.profile is None
                else {
                    "L": self.profile.L,
                    "D": self.profile.D,
                    "safe_4bit": self.profile.safe_4bit,
                }
            ),
        }

    def _log_load_summary(self):
        quant_desc = "4-bit NF4" if self.quant_config is not None else str(self.dtype)
        log.info("Loaded '%s' on %s (%s)", self.model_id, self.device, quant_desc)
        log.info(
            "Resolved %d layers | hidden_size=%s | profile=%s | output_hidden_states=%s",
            self.n_layers,
            self.hidden_size,
            "yes" if self.profile is not None else "no",
            self.output_hidden_states,
        )

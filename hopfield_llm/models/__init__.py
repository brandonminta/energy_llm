"""Model loading, profiles, and architecture resolution."""

from hopfield_llm.models.arch import (
    BACKBONE_PATHS,
    LAYER_ATTRS,
    resolve_backbone,
    resolve_layers,
)
from hopfield_llm.models.loader import HFLLM
from hopfield_llm.models.profiles import (
    DEFAULT_MODEL_ALIAS,
    MODEL_PROFILES,
    ModelProfile,
    resolve_model_profile,
)

__all__ = [
    "BACKBONE_PATHS",
    "DEFAULT_MODEL_ALIAS",
    "HFLLM",
    "LAYER_ATTRS",
    "MODEL_PROFILES",
    "ModelProfile",
    "resolve_backbone",
    "resolve_layers",
    "resolve_model_profile",
]

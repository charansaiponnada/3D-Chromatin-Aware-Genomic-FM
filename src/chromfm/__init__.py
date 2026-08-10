"""Chromatin-aware genomic foundation model."""

from chromfm.model import BiMambaLM, ModelConfig, build, use_scan  # noqa: F401

__all__ = ["BiMambaLM", "ModelConfig", "build", "use_scan"]

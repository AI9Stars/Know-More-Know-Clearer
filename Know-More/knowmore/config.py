"""Configuration loading.

One YAML file describes an entire run: which endpoint to talk to, which dataset
to read, where each stage writes its output. Stages never hardcode paths.

Resolution order, lowest priority first:

    1. dataclass defaults below
    2. the YAML file passed with ``--config``
    3. environment variables (endpoint settings only)
    4. ``--set key=value`` overrides on the command line

Paths in the YAML may use ``${data_root}`` and ``${run_root}``, which are
expanded against the corresponding top-level keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_INTERPOLATION = re.compile(r"\$\{(\w+)\}")


@dataclass
class EndpointConfig:
    """Where to reach the vLLM server (OpenAI-compatible API)."""

    base_url: str = "http://localhost:7999/v1"
    api_key: str = "EMPTY"
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 3

    def resolved(self) -> "EndpointConfig":
        """Apply ``VLLM_*`` environment overrides."""
        return EndpointConfig(
            base_url=os.getenv("VLLM_BASE_URL", self.base_url),
            api_key=os.getenv("VLLM_API_KEY", self.api_key),
            model=os.getenv("VLLM_MODEL", self.model),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )


@dataclass
class SamplingConfig:
    """Stage 1: repeated sampling to build the behavioral profile.

    ``n_samples`` is K, the number of responses drawn per query.
    """

    n_samples: int = 16
    max_tokens: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    workers: int = 8


@dataclass
class DiagnosisConfig:
    """Stage 2: meta-model judgement of knowledge status."""

    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    workers: int = 8


@dataclass
class MetaConfig:
    """Stage 3: meta-recognition and knowledge-query generation."""

    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    workers: int = 8
    max_samples_in_prompt: int = 6
    max_explanation_chars: int = 300


@dataclass
class SynthesisConfig:
    """Stage 5: region-conditioned QA synthesis."""

    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    workers: int = 8
    #: Retrieved contexts passed into the synthesis prompt per instance.
    max_passages: int = 5
    #: QA pairs mentioning any of these are discarded, so the generated data
    #: never leaks the retrieval scaffolding into the training targets.
    forbidden_terms: tuple = ("passage", "document", "the context")


@dataclass
class Config:
    """A full pipeline configuration."""

    data_root: str = "./data"
    run_root: str = "./runs/default"
    dataset: str = ""

    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    # Per-stage artifacts. Defaults are relative to run_root.
    input_path: str = ""
    sample_path: str = "${run_root}/1_samples.jsonl"
    diagnosis_path: str = "${run_root}/2_diagnosis.jsonl"
    meta_path: str = "${run_root}/3_meta.jsonl"
    retrieval_path: str = "${run_root}/4_retrieval.jsonl"
    synthesis_path: str = "${run_root}/5_synthesis.jsonl"
    sft_path: str = "${run_root}/6_train.jsonl"

    def path(self, name: str) -> Path:
        """Return an interpolated, absolute path for the named artifact."""
        raw = getattr(self, name)
        if not raw:
            raise ValueError(
                f"config key '{name}' is empty; set it in your YAML or pass --set {name}=..."
            )
        return Path(self._interpolate(raw)).expanduser().resolve()

    def _interpolate(self, value: str) -> str:
        def replace(match: re.Match) -> str:
            key = match.group(1)
            if key not in {"data_root", "run_root"}:
                raise ValueError(
                    f"unknown interpolation '${{{key}}}' (only data_root and run_root are supported)"
                )
            return str(getattr(self, key))

        return _INTERPOLATION.sub(replace, value)


def _build(cls, payload: Dict[str, Any]):
    """Instantiate a config dataclass, rejecting unknown keys loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**payload)


def load_config(path: Optional[str], overrides: Optional[Dict[str, str]] = None) -> Config:
    """Load ``path`` and apply ``--set`` overrides.

    ``overrides`` keys use dots for nesting, e.g. ``sampling.n_samples``.
    """
    payload: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

    nested = {
        "endpoint": EndpointConfig,
        "sampling": SamplingConfig,
        "diagnosis": DiagnosisConfig,
        "meta": MetaConfig,
        "synthesis": SynthesisConfig,
    }
    kwargs: Dict[str, Any] = {}
    for key, cls in nested.items():
        kwargs[key] = _build(cls, payload.pop(key, {}) or {})

    cfg = _build(Config, {**payload, **kwargs})

    for dotted, raw in (overrides or {}).items():
        _apply_override(cfg, dotted, raw)

    cfg.endpoint = cfg.endpoint.resolved()
    return cfg


def _apply_override(cfg: Config, dotted: str, raw: str) -> None:
    target: Any = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"--set {dotted}: no such section '{part}'")
        target = getattr(target, part)

    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"--set {dotted}: no such key '{leaf}'")

    current = getattr(target, leaf)
    setattr(target, leaf, _coerce(raw, type(current)))


def _coerce(raw: str, target_type: type) -> Any:
    if target_type is bool:
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot read '{raw}' as a boolean")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw

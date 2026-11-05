"""The six pipeline stages, in execution order.

Each module exposes ``run(cfg, ...)`` and reads the previous stage's output
path from the config, so stages compose without passing data in memory.
"""

from . import s1_sample, s2_diagnose, s3_meta, s4_retrieve, s5_synthesize, s6_build_sft

__all__ = ["s1_sample", "s2_diagnose", "s3_meta", "s4_retrieve", "s5_synthesize", "s6_build_sft"]

"""Know-More: meta-cognitive data construction for knowledge augmentation.

The data-construction half of *Know More, Know Clearer: A Meta-Cognitive
Framework for Knowledge Augmentation in Large Language Models*.

The pipeline probes a base model's knowledge boundary by repeated sampling,
partitions questions into mastered / confused / missing regions, and synthesizes
region-appropriate training data. The RL half lives in ``../Know-Clearer``.
"""

from .regions import Region, normalize_region

__version__ = "0.1.0"

__all__ = ["Region", "normalize_region", "__version__"]

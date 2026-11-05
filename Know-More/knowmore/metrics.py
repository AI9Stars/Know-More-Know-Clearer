"""Answer-matching used to turn a generation into a correctness flag.

The same normalization is used by the RL half of the project (Know-Clearer's
``qa_acc_reward.py``), so the correctness signal is identical on both sides.
"""

from __future__ import annotations

import re
import string
from typing import Sequence

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """Lowercase, strip all whitespace, and drop punctuation.

    Note this removes whitespace entirely rather than collapsing it, so
    ``"two"`` matches inside ``"twofold"``. That looseness is deliberate for a
    training signal but makes this unsuitable as an evaluation metric -- use
    exact match or token-level F1 when reporting final numbers.
    """
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    return text.translate(_PUNCTUATION)


def compute_accuracy(prediction: str, gold: str) -> float:
    """1.0 if the normalized gold answer appears in the normalized prediction."""
    return 1.0 if clean_text(gold) in clean_text(prediction) else 0.0


def mean_or_nan(values: Sequence[float]) -> float:
    """Mean of ``values``, or ``nan`` when empty -- avoids numpy warnings."""
    usable = [v for v in values if v == v]  # drop NaN
    return sum(usable) / len(usable) if usable else float("nan")

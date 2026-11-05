"""The knowledge-region taxonomy.

The model's knowledge space is partitioned into three regions via the
assignment mapping ``Gamma(B) -> r``:

    MASTERED  robust internal representations, high stability, low uncertainty
    CONFUSED  fragmented performance and fragile reasoning
    MISSING   a fundamental knowledge void; stochastic patterns, high uncertainty

Every stage keys off this taxonomy, so it lives in one module.

The region-assignment prompt asks the model to emit the labels
``"Knowledge Mastered"`` / ``"Knowledge Confused"`` / ``"Knowledge Missing"``;
:func:`normalize_region` parses those, the bare names, and the "feasibility"
spellings used by an earlier iteration of this codebase, so existing result
files stay readable.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Region(str, Enum):
    """A region of the model's knowledge space."""

    MASTERED = "mastered"
    CONFUSED = "confused"
    MISSING = "missing"

    @property
    def label(self) -> str:
        """The label emitted by the region-assignment prompt."""
        return _LABELS[self]

    @property
    def short(self) -> str:
        """Three-letter code used in filenames and log lines."""
        return _SHORT_CODES[self]

    @property
    def strategy(self) -> str:
        """The augmentation strategy this region calls for."""
        return _STRATEGIES[self]


_LABELS = {
    Region.MASTERED: "Knowledge Mastered",
    Region.CONFUSED: "Knowledge Confused",
    Region.MISSING: "Knowledge Missing",
}

_SHORT_CODES = {
    Region.MASTERED: "MAS",
    Region.CONFUSED: "CON",
    Region.MISSING: "MIS",
}

_STRATEGIES = {
    Region.MASTERED: "Boundary Expansion",
    Region.CONFUSED: "Structural Disambiguation",
    Region.MISSING: "Epistemic Foundation",
}

#: Spellings accepted on input: the prompt's labels, the bare region names, and
#: an older "feasibility" vocabulary kept so existing result files still load.
_ALIASES = {
    # labels emitted by the region-assignment prompt
    "knowledge mastered": Region.MASTERED,
    "knowledge confused": Region.CONFUSED,
    "knowledge missing": Region.MISSING,
    "knowledge_mastered": Region.MASTERED,
    "knowledge_confused": Region.CONFUSED,
    "knowledge_missing": Region.MISSING,
    # legacy knowledge_status vocabulary
    "partially_mastered": Region.CONFUSED,
    "unmastered": Region.MISSING,
    # legacy feasibility vocabulary
    "fully_feasible": Region.MASTERED,
    "partially_feasible": Region.CONFUSED,
    "fully_infeasible": Region.MISSING,
    "ff": Region.MASTERED,
    "pf": Region.CONFUSED,
    "pi": Region.CONFUSED,  # an early typo for PF; present in some result files
    "fi": Region.MISSING,
    # current short codes
    "mas": Region.MASTERED,
    "con": Region.CONFUSED,
    "mis": Region.MISSING,
}


def normalize_region(value: object) -> Optional[Region]:
    """Coerce ``value`` into a :class:`Region`, accepting every known spelling.

    Returns ``None`` when the value is missing or unrecognized, so callers can
    decide whether to skip the record or fall back.
    """
    if isinstance(value, Region):
        return value
    if not isinstance(value, str):
        return None

    key = value.strip().lower()
    if not key:
        return None
    try:
        return Region(key)
    except ValueError:
        pass
    return _ALIASES.get(key)

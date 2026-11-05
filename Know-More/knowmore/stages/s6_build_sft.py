"""Stage 6 -- flatten the augmented instances into training data.

The SFT objective is over cognitively-transformed instances (q', a'), so a
training row is a question and a short answer. Rule 4 of the generation prompts
forbids reasoning steps, so there is nothing else to carry.

Two output shapes, for the two consumers:

``sft``
    ``{"question", "answer", "meta_query", "meta_answer", "region"}`` -- the
    training pair plus provenance, so you can slice the SFT set by region or
    trace a row back to the seed instance it was derived from.

``qa``
    ``{"question", "answer"}`` only. This is what the calibration half
    (Know-Clearer) converts to parquet for GRPO; see
    ``Know-Clearer/verl/data/data_format.py``.

Input   stage 5 output
Output  one line per QA pair
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

from ..config import Config
from ..io_utils import load_jsonl, require_input, write_jsonl
from ..regions import Region, normalize_region

log = logging.getLogger(__name__)

FORMATS = ("sft", "qa")


def _flatten(records: List[Dict[str, Any]], fmt: str) -> Iterator[Dict[str, Any]]:
    for record in records:
        region = normalize_region(record.get("region"))
        seed_question = record.get("question", "")
        seed_answer = record.get("gold_answer", "")

        for qa in record.get("qa_pairs", []) or []:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue

            if fmt == "qa":
                yield {"question": question, "answer": answer}
            else:
                yield {
                    "question": question,
                    "answer": answer,
                    "meta_query": seed_question,
                    "meta_answer": seed_answer,
                    "region": region.value if region else "",
                }


def run(cfg: Config, fmt: str = "sft") -> None:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}'; expected one of {', '.join(FORMATS)}")

    in_path = cfg.path("synthesis_path")
    out_path = cfg.path("sft_path")
    require_input(in_path, stage="build", produced_by="knowmore synthesize")

    records = load_jsonl(in_path)
    rows = list(_flatten(records, fmt))
    n = write_jsonl(out_path, rows)

    log.info("flattened %d records into %d '%s' rows -> %s", len(records), n, fmt, out_path)

    if fmt == "sft":
        per_region: Dict[str, int] = {}
        for row in rows:
            per_region[row["region"]] = per_region.get(row["region"], 0) + 1
        for region in Region:
            count = per_region.get(region.value, 0)
            log.info("  %-9s %6d rows (%5.1f%%)", region.value, count,
                     100 * count / n if n else 0.0)
    else:
        log.info(
            "convert to parquet for the calibration stage with:\n"
            "  python Know-Clearer/verl/data/data_format.py "
            "--input %s --output <out>.parquet --split train",
            out_path,
        )

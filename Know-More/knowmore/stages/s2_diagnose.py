"""Stage 2 -- meta-cognitive region assignment.

The model reads its own behavioral profile B(q, a) -- all K sampled responses
with their correctness and uncertainty -- and assigns the instance to one of the
three knowledge regions. This is the "monitoring" half of meta-cognition:
the model's intrinsic evaluative capacity is what performs the partition.

The prompt states a performance-oriented principle: accuracy and uncertainty are
reference signals, but the observed answer pattern decides. When the model
returns something unusable, a deterministic accuracy-threshold fallback keeps
the record rather than dropping it; every fallback is tagged
``assignment_source: "fallback"`` so it can be audited or excluded.

Input   stage 1 output
Output  the same records plus ``region``, ``knowledge_status``, ``explanation``,
        ``assignment_source``
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..config import Config
from ..io_utils import JSONLWriter, completed_keys, load_jsonl, require_input
from ..llm import LLMClient
from ..prompts import ACCURACY_HIGH, ACCURACY_LOW, REGION_ASSIGNMENT
from ..regions import Region, normalize_region

log = logging.getLogger(__name__)


def _samples_json(samples: List[Dict[str, Any]]) -> str:
    """One JSON object per line: response, correctness, uncertainty."""
    return "\n".join(
        json.dumps(
            {
                "sample_id": s.get("sample_id"),
                "response": s.get("response"),
                "accuracy": s.get("accuracy", 0),
                "uncertainty": s.get("uncertainty", s.get("avg_surprisal")),
            },
            ensure_ascii=False,
        )
        for s in samples
    )


def build_prompt(record: Dict[str, Any]) -> str:
    mean_acc = _as_float(record.get("mean_acc"))
    mean_unc = _as_float(_uncertainty_of(record))
    return REGION_ASSIGNMENT.format(
        question=record.get("question", ""),
        gold_answer=record.get("gold_answer", ""),
        mean_accuracy=f"{mean_acc:.4f}" if mean_acc == mean_acc else "unknown",
        mean_uncertainty=f"{mean_unc:.4f}" if mean_unc == mean_unc else "unknown",
        samples_json=_samples_json(record.get("samples", [])),
    )


def assign_region(client: LLMClient, cfg: Config, record: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the model to assign a region, falling back on failure."""
    try:
        raw = client.complete_json(
            build_prompt(record),
            max_tokens=cfg.diagnosis.max_tokens,
            temperature=cfg.diagnosis.temperature,
            top_p=cfg.diagnosis.top_p,
        )
    except Exception as exc:  # noqa: BLE001 - fall back rather than lose the record
        return fallback_assignment(record, exc)

    region = normalize_region(raw.get("Knowledge_Status"))
    if region is None:
        return fallback_assignment(record, ValueError("unparseable Knowledge_Status"), raw)

    return {
        "region": region.value,
        "knowledge_status": region.label,
        "explanation": str(raw.get("Explanation", "")).strip(),
        "assignment_source": "meta_model",
    }


def fallback_assignment(
    record: Dict[str, Any],
    error: Optional[Exception] = None,
    raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic classification on the prompt's reference boundaries.

    This applies the stated thresholds without the +/-0.10 tolerance, which
    depends on judging answer coherence. It is a floor, not a substitute for
    the model's assignment.
    """
    mean_acc = _as_float(record.get("mean_acc"))

    if mean_acc != mean_acc:  # NaN
        region = Region.MISSING
        reason = "mean_acc is missing"
    elif mean_acc >= ACCURACY_HIGH:
        region = Region.MASTERED
        reason = f"mean_acc={mean_acc:.2f} >= {ACCURACY_HIGH}"
    elif mean_acc <= ACCURACY_LOW:
        region = Region.MISSING
        reason = f"mean_acc={mean_acc:.2f} <= {ACCURACY_LOW}"
    else:
        region = Region.CONFUSED
        reason = f"{ACCURACY_LOW} < mean_acc={mean_acc:.2f} < {ACCURACY_HIGH}"

    explanation = f"Fallback assignment on the reference accuracy boundary: {reason}."
    if error is not None:
        explanation += f" Meta-model error: {type(error).__name__}: {error}"

    result = {
        "region": region.value,
        "knowledge_status": region.label,
        "explanation": explanation,
        "assignment_source": "fallback",
    }
    if raw is not None:
        result["meta_raw"] = raw
    return result


def _uncertainty_of(record: Dict[str, Any]) -> Any:
    """Read mean uncertainty, tolerating the legacy ``mean_surprisal`` name."""
    if "mean_uncertainty" in record:
        return record["mean_uncertainty"]
    return record.get("mean_surprisal")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run(cfg: Config, resume: bool = False) -> None:
    in_path = cfg.path("sample_path")
    out_path = cfg.path("diagnosis_path")
    require_input(in_path, stage="diagnose", produced_by="knowmore sample")

    records = load_jsonl(in_path)
    done = completed_keys(out_path, "question") if resume else set()
    pending = [r for r in records if str(r.get("question", "")) not in done]
    if not pending:
        log.info("nothing to do; all %d records already assigned", len(records))
        return

    client = LLMClient(cfg.endpoint)
    counts: Dict[str, int] = {}
    fallbacks = 0

    with JSONLWriter(out_path, resume=resume) as writer:
        with ThreadPoolExecutor(max_workers=cfg.diagnosis.workers) as pool:
            results = pool.map(lambda r: (r, assign_region(client, cfg, r)), pending)
            for record, assignment in tqdm(
                results, total=len(pending), desc="Assigning regions", ncols=100
            ):
                counts[assignment["region"]] = counts.get(assignment["region"], 0) + 1
                fallbacks += assignment["assignment_source"] == "fallback"
                writer.write({**record, **assignment})

    total = sum(counts.values())
    log.info("assigned %d records -> %s", total, out_path)
    for region in Region:
        n = counts.get(region.value, 0)
        log.info("  %-9s %6d (%5.1f%%)  %s", region.value, n,
                 100 * n / total if total else 0.0, region.strategy)
    if fallbacks:
        log.warning(
            "%d/%d records (%.1f%%) used the threshold fallback; filter on "
            "assignment_source == 'fallback' to inspect them",
            fallbacks, total, 100 * fallbacks / total,
        )

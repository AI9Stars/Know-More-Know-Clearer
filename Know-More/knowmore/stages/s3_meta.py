"""Stage 3 -- cognitive tagging and search-query generation.

Cognitive Tags are the critical entities, logical prerequisites, or core
concepts that act as the primary subjects of the identified knowledge state.
They serve as the semantic anchors for formulating a high-precision search
query.

The strategy is conditioned on the assigned region -- retrieve prerequisites for
MISSING, disambiguating links for CONFUSED, adjacent material for MASTERED --
but all three are expressed inside a single prompt, which reads the region from
the upstream diagnosis rather than being selected by it.

Input   stage 2 output
Output  the same records plus ``knowledge_query``
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from tqdm import tqdm

from ..config import Config
from ..io_utils import JSONLWriter, completed_keys, load_jsonl, require_input
from ..llm import LLMClient
from ..prompts import SEARCH_QUERY
from ..regions import Region, normalize_region

log = logging.getLogger(__name__)


def _truncate(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    if not text:
        return "(no explanation)"
    return text if len(text) <= limit else text[:limit] + "..."


def build_prompt(cfg: Config, record: Dict[str, Any], region: Region) -> str:
    return SEARCH_QUERY.format(
        question=record.get("question", ""),
        gold_answer=record.get("gold_answer", ""),
        knowledge_status=region.label,
        explanation=_truncate(record.get("explanation"), cfg.meta.max_explanation_chars),
    )


def generate_query(
    client: LLMClient, cfg: Config, record: Dict[str, Any], region: Region
) -> Dict[str, Any]:
    """Return the retrieval query for one record; empty string on failure."""
    try:
        raw = client.complete_json(
            build_prompt(cfg, record, region),
            max_tokens=cfg.meta.max_tokens,
            temperature=cfg.meta.temperature,
            top_p=cfg.meta.top_p,
        )
    except Exception as exc:  # noqa: BLE001 - keep the record, flag the failure
        log.debug("query generation failed for %r: %s", str(record.get("question"))[:60], exc)
        return {"knowledge_query": "", "query_error": f"{type(exc).__name__}: {exc}"}

    return {"knowledge_query": str(raw.get("Search_Query", "")).strip(), "query_error": ""}


def run(cfg: Config, resume: bool = False) -> None:
    in_path = cfg.path("diagnosis_path")
    out_path = cfg.path("meta_path")
    require_input(in_path, stage="meta", produced_by="knowmore diagnose")

    records, unknown = [], 0
    for record in load_jsonl(in_path):
        region = normalize_region(record.get("region"))
        if region is None:
            unknown += 1
            continue
        records.append((record, region))
    if unknown:
        log.warning("%d records have an unrecognized region and were skipped", unknown)

    done = completed_keys(out_path, "question") if resume else set()
    pending = [(r, g) for r, g in records if str(r.get("question", "")) not in done]
    if not pending:
        log.info("nothing to do; all %d records already have a query", len(records))
        return

    client = LLMClient(cfg.endpoint)
    failures = 0
    backfilled = 0

    def work(item):
        record, region = item
        return record, generate_query(client, cfg, record, region)

    with JSONLWriter(out_path, resume=resume) as writer:
        with ThreadPoolExecutor(max_workers=cfg.meta.workers) as pool:
            for record, result in tqdm(
                pool.map(work, pending), total=len(pending), desc="Cognitive tagging", ncols=100
            ):
                failures += bool(result["query_error"])
                # Retrieval needs a query for every record. When generation
                # produced none, fall back to the original question so the
                # instance stays in the pipeline instead of silently vanishing.
                if not result["knowledge_query"]:
                    result["knowledge_query"] = str(record.get("question", "")).strip()
                    result["query_source"] = "original_question"
                    backfilled += 1
                else:
                    result["query_source"] = "meta_model"
                writer.write({**record, **result})

    log.info("generated %d queries -> %s", len(pending), out_path)
    if failures:
        log.warning("%d/%d query calls failed (see query_error)", failures, len(pending))
    if backfilled:
        log.warning(
            "%d/%d records fell back to the original question as the retrieval "
            "query, which skips the cognitive-tagging step for those instances",
            backfilled, len(pending),
        )

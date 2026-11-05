"""Stage 5 -- region-specific data augmentation.

Each region gets its own generation prompt, implementing the three strategies:

    MISSING   Epistemic Foundation -- treat the passages as the authoritative
              source and synthesize the missing definitions and factual pillars
    CONFUSED  Structural Disambiguation -- clarify and decompose, anchoring the
              reasoning to explicit evidence
    MASTERED  Boundary Expansion -- extend to comparisons and broader context
              while preserving the established structure

Per Rule 4 of all three templates, generated pairs are ``(question, answer)``
with no reasoning steps: the SFT objective is over transformed instances
(q', a').

Input   stage 4 output
Output  the same records plus ``qa_pairs``
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from tqdm import tqdm

from ..config import Config
from ..io_utils import JSONLWriter, completed_keys, load_jsonl, require_input
from ..llm import LLMClient
from ..prompts import QA_GENERATION
from ..regions import Region, normalize_region

log = logging.getLogger(__name__)


def format_passages(passages: List[str], max_passages: int) -> str:
    """Number the passages so the model can refer to them internally."""
    if not passages:
        return "(no retrieved passages provided)"
    return "\n".join(f"{i}. {p}" for i, p in enumerate(passages[:max_passages], 1))


def _is_usable(qa: Any, forbidden: tuple) -> bool:
    """Keep only complete pairs that do not leak the retrieval scaffolding.

    Rule 4 forbids reasoning steps and the prompts never mention passages, but
    models slip; dropping the offenders keeps that noise out of the SFT data.
    """
    if not isinstance(qa, dict):
        return False
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    if not question or not answer:
        return False
    blob = f"{question} {answer}".lower()
    return not any(term in blob for term in forbidden)


def synthesize(
    client: LLMClient, cfg: Config, record: Dict[str, Any], region: Region
) -> List[Dict[str, str]]:
    prompt = QA_GENERATION[region].format(
        query=str(record.get("knowledge_query") or record.get("question", "")).strip(),
        retrieved_passages=format_passages(
            record.get("retrieved_passages") or [], cfg.synthesis.max_passages
        ),
    )

    raw = client.chat_json(
        [{"role": "user", "content": prompt}],
        max_tokens=cfg.synthesis.max_tokens,
        temperature=cfg.synthesis.temperature,
        top_p=cfg.synthesis.top_p,
    )

    pairs = raw.get("qa_pairs", [])
    if not isinstance(pairs, list):
        return []

    return [
        {"question": str(qa["question"]).strip(), "answer": str(qa["answer"]).strip()}
        for qa in pairs
        if _is_usable(qa, cfg.synthesis.forbidden_terms)
    ]


def run(cfg: Config, resume: bool = False) -> None:
    in_path = cfg.path("retrieval_path")
    out_path = cfg.path("synthesis_path")
    require_input(in_path, stage="synthesize", produced_by="knowmore retrieve")

    records = load_jsonl(in_path)
    done = completed_keys(out_path, "question") if resume else set()

    pending, unknown = [], 0
    for record in records:
        if str(record.get("question", "")) in done:
            continue
        region = normalize_region(record.get("region"))
        if region is None:
            unknown += 1
            continue
        pending.append((record, region))

    if unknown:
        log.warning("%d records have an unrecognized region and were skipped", unknown)
    if not pending:
        log.info("nothing to do; all %d records already synthesized", len(records))
        return

    client = LLMClient(cfg.endpoint)
    per_region: Dict[str, int] = {}
    total_pairs = 0
    empty = 0

    def work(item):
        record, region = item
        try:
            return record, region, synthesize(client, cfg, record, region)
        except Exception as exc:  # noqa: BLE001 - one bad record must not kill the run
            log.debug("synthesis failed for %r: %s", str(record.get("question"))[:60], exc)
            return record, region, []

    with JSONLWriter(out_path, resume=resume) as writer:
        with ThreadPoolExecutor(max_workers=cfg.synthesis.workers) as pool:
            for record, region, pairs in tqdm(
                pool.map(work, pending), total=len(pending), desc="Synthesizing", ncols=100
            ):
                if not pairs:
                    empty += 1
                    continue
                per_region[region.value] = per_region.get(region.value, 0) + len(pairs)
                total_pairs += len(pairs)
                writer.write({**record, "region_short": region.short, "qa_pairs": pairs})

    log.info("synthesized %d QA pairs -> %s", total_pairs, out_path)
    for region in Region:
        log.info("  %-9s %6d pairs  (%s)", region.value,
                 per_region.get(region.value, 0), region.strategy)
    if empty:
        log.warning(
            "%d/%d records produced no usable pairs (generation failed or every "
            "pair was filtered)", empty, len(pending),
        )

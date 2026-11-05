"""Stage 4 -- attach retrieved passages to each knowledge query.

Retrieval itself is done by an external engine. Our runs used UltraRAG
(https://github.com/OpenBMB/UltraRAG) over a Wikipedia corpus, but nothing here
is UltraRAG-specific: this stage only needs a mapping from query text to a list
of passage strings.

Because the retriever runs out-of-process, this stage has two halves:

    knowmore retrieve --export queries.jsonl
        write the knowledge queries for your retriever to consume

    knowmore retrieve --results <engine output>
        merge the engine's output back onto the records

Input   stage 3 output
Output  the same records plus ``retrieved_passages``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..io_utils import load_jsonl, require_input, write_jsonl

log = logging.getLogger(__name__)

#: Keys under which retrieval engines commonly nest the passage text.
_PASSAGE_KEYS = ("contents", "content", "text", "passage", "segment")
#: Keys under which they nest the query.
_QUERY_KEYS = ("query", "question", "knowledge_query")


def export_queries(cfg: Config, dest: Path) -> None:
    """Write ``{"query": ...}`` lines for the retriever to read."""
    in_path = cfg.path("meta_path")
    require_input(in_path, stage="retrieve", produced_by="knowmore meta")

    records = load_jsonl(in_path)
    seen, rows = set(), []
    for record in records:
        query = str(record.get("knowledge_query", "")).strip()
        if query and query not in seen:
            seen.add(query)
            rows.append({"query": query})

    n = write_jsonl(dest, rows)
    log.info("exported %d unique queries (from %d records) to %s", n, len(records), dest)
    log.info("run your retriever over this file, then re-run with --results <engine output>")


def _extract_passages(entry: Any, top_k: Optional[int]) -> List[str]:
    """Pull passage strings out of one engine result entry."""
    raw = entry
    if isinstance(entry, dict):
        for key in ("passages", "results", "retrieved", "documents", "docs"):
            if key in entry:
                raw = entry[key]
                break

    if not isinstance(raw, list):
        return []

    passages: List[str] = []
    for item in raw:
        if isinstance(item, str):
            passages.append(item)
        elif isinstance(item, dict):
            for key in _PASSAGE_KEYS:
                if isinstance(item.get(key), str):
                    passages.append(item[key])
                    break
    return passages[:top_k] if top_k else passages


def _build_index(payload: Any, top_k: Optional[int]) -> Dict[str, List[str]]:
    """Map query text -> passages, accepting the common engine output shapes."""
    index: Dict[str, List[str]] = {}

    if isinstance(payload, dict):
        # Either {query: [passages]} or {"results": [...]}.
        inner = payload.get("results", payload)
        if isinstance(inner, dict):
            for query, value in inner.items():
                index[str(query).strip()] = _extract_passages(value, top_k)
            return index
        payload = inner

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            query = next(
                (str(entry[k]).strip() for k in _QUERY_KEYS if isinstance(entry.get(k), str)),
                None,
            )
            if query:
                index[query] = _extract_passages(entry, top_k)

    return index


def merge_results(cfg: Config, results_path: Path, top_k: Optional[int] = None) -> None:
    """Join engine output onto the stage 3 records by query text."""
    in_path = cfg.path("meta_path")
    out_path = cfg.path("retrieval_path")
    require_input(in_path, stage="retrieve", produced_by="knowmore meta")
    require_input(results_path, stage="retrieve")

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    index = _build_index(payload, top_k)
    if not index:
        raise ValueError(
            f"could not read any query->passage pairs from {results_path}. "
            "Expected a list of {query, passages} objects or a {query: passages} mapping."
        )
    log.info("loaded passages for %d queries from %s", len(index), results_path)

    records = load_jsonl(in_path)
    matched = 0
    for record in records:
        query = str(record.get("knowledge_query", "")).strip()
        passages = index.get(query, [])
        record["retrieved_passages"] = passages
        matched += bool(passages)

    write_jsonl(out_path, records)
    log.info("wrote %d records to %s", len(records), out_path)
    if matched < len(records):
        log.warning(
            "%d/%d records got no passages; they will still be synthesized but "
            "without retrieved evidence",
            len(records) - matched,
            len(records),
        )


def run(cfg: Config, results: Optional[str] = None, export: Optional[str] = None, top_k: Optional[int] = None) -> None:
    if export:
        export_queries(cfg, Path(export).expanduser().resolve())
        return
    if results:
        merge_results(cfg, Path(results).expanduser().resolve(), top_k)
        return
    raise ValueError(
        "stage 'retrieve' needs either --export <path> (to dump queries for your "
        "retriever) or --results <path> (to merge the retriever's output back in)."
    )

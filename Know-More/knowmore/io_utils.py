"""JSONL helpers shared by every stage.

The pipeline runs for hours over tens of thousands of questions against a local
server, so two properties matter: a crashed stage must not lose completed work,
and re-running a stage must not redo it. Both are handled here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

log = logging.getLogger(__name__)


def read_jsonl(path: Path, skip_bad: bool = True) -> Iterator[Dict[str, Any]]:
    """Yield objects from a JSONL file.

    Malformed lines are logged and skipped by default rather than killing a
    long run; pass ``skip_bad=False`` to raise instead.
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not skip_bad:
                    raise
                log.warning("%s:%d is not valid JSON, skipping", path, lineno)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write ``records``, creating parent directories. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


class JSONLWriter:
    """Append-mode writer that flushes each record.

    Flushing per record costs little next to an LLM call and means a killed run
    keeps everything it had finished.
    """

    def __init__(self, path: Path, resume: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a" if resume else "w", encoding="utf-8")
        self.count = 0

    def write(self, record: Dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def completed_keys(path: Path, key: str = "question") -> Set[str]:
    """Return the ``key`` values already present in ``path``.

    Used to skip finished work when resuming. A missing file yields an empty
    set, so the first run needs no special case.
    """
    if not path.exists():
        return set()
    done = {str(record[key]) for record in read_jsonl(path) if key in record}
    if done:
        log.info("resuming: %d records already in %s", len(done), path.name)
    return done


def require_input(path: Path, stage: str, produced_by: Optional[str] = None) -> None:
    """Fail early with an actionable message when a stage's input is absent."""
    if path.exists():
        return
    hint = f" Run `{produced_by}` first." if produced_by else ""
    raise FileNotFoundError(f"stage '{stage}' needs {path}, which does not exist.{hint}")

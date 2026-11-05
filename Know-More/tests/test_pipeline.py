"""Unit tests for the pieces that do not need a live endpoint.

Run with `python -m pytest tests/` or `python tests/test_pipeline.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowmore.config import load_config
from knowmore.io_utils import completed_keys, load_jsonl, write_jsonl
from knowmore.llm import JSONParseError, extract_json
from knowmore.metrics import clean_text, compute_accuracy, mean_or_nan
from knowmore.prompts import QA_GENERATION, REGION_ASSIGNMENT, SAMPLING_PROMPT, SEARCH_QUERY
from knowmore.regions import Region, normalize_region
from knowmore.stages import s2_diagnose, s4_retrieve, s6_build_sft

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


# --------------------------------------------------------------------------
print("\n[regions]")
check("paper names are the canonical values",
      [r.value for r in Region] == ["mastered", "confused", "missing"])
check("legacy fully_feasible -> mastered", normalize_region("fully_feasible") is Region.MASTERED)
check("legacy partially_feasible -> confused", normalize_region("partially_feasible") is Region.CONFUSED)
check("legacy fully_infeasible -> missing", normalize_region("fully_infeasible") is Region.MISSING)
check("legacy short code FF", normalize_region("FF") is Region.MASTERED)
check("legacy PI typo maps to confused", normalize_region("PI") is Region.CONFUSED)
check("current names round-trip", normalize_region("missing") is Region.MISSING)
check("unknown region -> None", normalize_region("banana") is None)
check("empty -> None", normalize_region("") is None)
check("None -> None", normalize_region(None) is None)
check("legacy partially_mastered -> confused", normalize_region("partially_mastered") is Region.CONFUSED)
check("paper label Knowledge Mastered", normalize_region("Knowledge Mastered") is Region.MASTERED)
check("paper label Knowledge Confused", normalize_region("Knowledge Confused") is Region.CONFUSED)
check("paper label Knowledge Missing", normalize_region("Knowledge Missing") is Region.MISSING)
check("labels round-trip", all(normalize_region(r.label) is r for r in Region))
check("strategies distinct", len({r.strategy for r in Region}) == 3)
check("short codes distinct", len({r.short for r in Region}) == 3)


# --------------------------------------------------------------------------
print("\n[metrics]")
check("substring match", compute_accuracy("The answer is Paris.", "Paris") == 1.0)
check("wrong answer", compute_accuracy("I don't know", "Paris") == 0.0)
check("case/punct insensitive", compute_accuracy("paris!", "Paris") == 1.0)
check("whitespace stripped", clean_text("New  York") == "newyork")
check("known looseness: two in twofold", compute_accuracy("twofold increase", "two") == 1.0)
check("mean ignores NaN", mean_or_nan([1.0, float("nan"), 0.0]) == 0.5)
check("mean of empty is NaN", mean_or_nan([]) != mean_or_nan([]))


# --------------------------------------------------------------------------
print("\n[prompts]")
for region in Region:
    check(f"QA prompt exists for {region.value}", region in QA_GENERATION)
    _qa = QA_GENERATION[region].format(query="Q", retrieved_passages="P")
    check(f"{region.value} QA prompt formats", "{query}" not in _qa and "Q" in _qa)
    check(f"{region.value} QA prompt emits qa_pairs", '"qa_pairs"' in _qa)
    check(f"{region.value} QA prompt forbids reasoning",
          "Do not include reasoning steps" in _qa)
    check(f"{region.value} QA prompt has no thought field", '"thought"' not in _qa)
    check(f"{region.value} QA prompt caps answer length", "20 words" in _qa)

check("sampling prompt matches paper wording",
      SAMPLING_PROMPT.format(question="X")
      == "Please provide the user with the answer directly. Question: X \nAnswer:")
check("sampling prompt has no CoT clause", "reasoning process" not in SAMPLING_PROMPT)

_assign = REGION_ASSIGNMENT.format(
    question="q", gold_answer="a", mean_accuracy="0.5",
    mean_uncertainty="1.0", samples_json="{}")
check("REGION_ASSIGNMENT formats", "{question}" not in _assign)
check("REGION_ASSIGNMENT emits Knowledge_Status", '"Knowledge_Status"' in _assign)
check("REGION_ASSIGNMENT has no error_type", "error_type" not in _assign.lower())
check("REGION_ASSIGNMENT uses Uncertainty not surprisal",
      "Mean Uncertainty" in _assign and "surprisal" not in _assign.lower())
check("REGION_ASSIGNMENT states both boundaries",
      ">= 0.80" in _assign and "<= 0.20" in _assign)

_sq = SEARCH_QUERY.format(question="q", gold_answer="a",
                          knowledge_status="Knowledge Missing", explanation="e")
check("SEARCH_QUERY formats", "{question}" not in _sq)
check("SEARCH_QUERY emits Search_Query only", '"Search_Query"' in _sq)
check("SEARCH_QUERY has Cognitive Tags", "[Cognitive Tags]" in _sq)
check("SEARCH_QUERY dropped Meta-Recognition", "Meta-Recognition" not in _sq)
check("SEARCH_QUERY covers all three strategies",
      all(f"Knowledge {n} Strategy" in _sq for n in ("Missing", "Confused", "Mastered")))

print("\n[llm.extract_json]")
check("bare object", extract_json('{"a": 1}') == {"a": 1})
check("fenced", extract_json('```json\n{"a": 2}\n```') == {"a": 2})
check("prose wrapped", extract_json('Sure! {"a": 3} hope that helps') == {"a": 3})
check("nested braces", extract_json('x {"a": {"b": 4}} y') == {"a": {"b": 4}})
try:
    extract_json("no json here")
    check("raises on garbage", False)
except JSONParseError:
    check("raises on garbage", True)


# --------------------------------------------------------------------------
print("\n[region assignment + fallback]")
_ok = s2_diagnose.assign_region.__doc__ is not None
check("assign_region documented", _ok)

for label, expected in [("Knowledge Mastered", "mastered"),
                        ("Knowledge Confused", "confused"),
                        ("Knowledge Missing", "missing")]:
    r = normalize_region(label)
    check(f"parse {label!r}", r is not None and r.value == expected)

for acc, expected in [(0.95, "mastered"), (0.80, "mastered"), (0.5, "confused"),
                      (0.20, "missing"), (0.05, "missing")]:
    fb = s2_diagnose.fallback_assignment({"mean_acc": acc})
    check(f"fallback acc={acc} -> {expected}", fb["region"] == expected, f"got {fb['region']}")
    check(f"fallback acc={acc} tagged", fb["assignment_source"] == "fallback")
    check(f"fallback acc={acc} sets paper label",
          fb["knowledge_status"] == normalize_region(expected).label)

_nan_fb = s2_diagnose.fallback_assignment({})
check("fallback handles missing mean_acc", _nan_fb["region"] == "missing")

_rec = {"question": "q", "gold_answer": "a", "mean_acc": 0.5,
        "mean_surprisal": 1.25, "samples": [{"sample_id": 0, "response": "r", "accuracy": 1}]}
_built = s2_diagnose.build_prompt(_rec)
check("build_prompt reads legacy mean_surprisal", "1.2500" in _built)
_rec2 = dict(_rec); _rec2["mean_uncertainty"] = 2.5
check("mean_uncertainty preferred over legacy", "2.5000" in s2_diagnose.build_prompt(_rec2))

print("\n[retrieval result parsing]")
check("list of {query, passages}",
      s4_retrieve._build_index([{"query": "q1", "passages": ["p1", "p2"]}], None) == {"q1": ["p1", "p2"]})
check("dict mapping",
      s4_retrieve._build_index({"q1": ["p1"]}, None) == {"q1": ["p1"]})
check("passages as dicts with contents",
      s4_retrieve._build_index([{"query": "q", "passages": [{"contents": "c"}]}], None) == {"q": ["c"]})
check("nested under results key",
      s4_retrieve._build_index({"results": [{"question": "q", "docs": ["d"]}]}, None) == {"q": ["d"]})
check("top_k truncates",
      s4_retrieve._build_index([{"query": "q", "passages": ["a", "b", "c"]}], 2) == {"q": ["a", "b"]})
check("unparseable yields empty", s4_retrieve._build_index(42, None) == {})


# --------------------------------------------------------------------------
print("\n[sft flattening]")
_records = [{
    "question": "source q", "gold_answer": "source a", "region": "fully_feasible",
    "qa_pairs": [
        {"question": "q1", "answer": "a1"},
        {"question": "", "answer": "a2"},      # dropped: no question
        {"question": "q3", "answer": ""},      # dropped: no answer
        {"question": "q4", "answer": "a4"},
    ],
}]
_sft = list(s6_build_sft._flatten(_records, "sft"))
check("incomplete pairs dropped", len(_sft) == 2, f"got {len(_sft)}")
check("answer is the bare short answer", _sft[0]["answer"] == "a1")
check("no think wrapper", "<think>" not in _sft[0]["answer"])
check("legacy region normalized in output", _sft[0]["region"] == "mastered")
check("provenance kept", _sft[0]["meta_query"] == "source q")

_qa = list(s6_build_sft._flatten(_records, "qa"))
check("qa format is 2 keys only", set(_qa[0]) == {"question", "answer"})
check("qa matches sft answer", _qa[0]["answer"] == _sft[0]["answer"])

print("\n[config]")
with tempfile.TemporaryDirectory() as td:
    cfg_path = Path(td) / "c.yaml"
    cfg_path.write_text(
        "data_root: /tmp/d\n"
        "run_root: /tmp/r\n"
        "input_path: ${data_root}/in.jsonl\n"
        "endpoint:\n  model: /models/x\n"
        "sampling:\n  n_samples: 4\n"
    )
    cfg = load_config(str(cfg_path))
    check("interpolates data_root", str(cfg.path("input_path")) == "/tmp/d/in.jsonl")
    check("interpolates run_root", str(cfg.path("sample_path")) == "/tmp/r/1_samples.jsonl")
    check("nested value read", cfg.sampling.n_samples == 4)
    check("default preserved", cfg.sampling.max_tokens == 32)

    cfg2 = load_config(str(cfg_path), {"sampling.n_samples": "9", "run_root": "/tmp/z"})
    check("override scalar", cfg2.sampling.n_samples == 9)
    check("override is typed as int", isinstance(cfg2.sampling.n_samples, int))
    check("override affects interpolation", str(cfg2.path("sample_path")) == "/tmp/z/1_samples.jsonl")

    try:
        load_config(str(cfg_path), {"sampling.nope": "1"})
        check("unknown override rejected", False)
    except ValueError:
        check("unknown override rejected", True)

    bad = Path(td) / "bad.yaml"
    bad.write_text("nonsense_key: 1\n")
    try:
        load_config(str(bad))
        check("unknown config key rejected", False)
    except ValueError:
        check("unknown config key rejected", True)


# --------------------------------------------------------------------------
print("\n[io_utils]")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "nested" / "f.jsonl"
    write_jsonl(p, [{"question": "a"}, {"question": "b"}])
    check("creates parent dirs", p.exists())
    check("round-trips", [r["question"] for r in load_jsonl(p)] == ["a", "b"])
    check("completed_keys", completed_keys(p, "question") == {"a", "b"})
    check("completed_keys on missing file", completed_keys(Path(td) / "nope.jsonl") == set())

    corrupt = Path(td) / "corrupt.jsonl"
    corrupt.write_text('{"question": "ok"}\nNOT JSON\n{"question": "ok2"}\n')
    check("bad lines skipped", len(load_jsonl(corrupt)) == 2)


# --------------------------------------------------------------------------
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL TESTS PASSED")

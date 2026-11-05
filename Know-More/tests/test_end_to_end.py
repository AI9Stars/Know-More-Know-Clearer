"""End-to-end run of all six stages against a stubbed endpoint.

No GPU and no server: :class:`FakeClient` stands in for the vLLM endpoint and
returns canned replies shaped like the ones the prompts ask for. This exercises
the real stage code, the real config plumbing, and the real file handoffs, so it
catches wiring breaks between stages.

Run with `python tests/test_end_to_end.py`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowmore import llm
from knowmore.config import load_config
from knowmore.io_utils import load_jsonl
from knowmore.stages import s1_sample, s2_diagnose, s3_meta, s4_retrieve, s5_synthesize, s6_build_sft

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


QUESTIONS = [
    # (question, gold, how often the fake model gets it right) -> region
    ("who discovered radium", "Marie Curie", 1.0),      # -> mastered
    ("when was the treaty signed", "1783", 0.5),        # -> confused
    ("what is the airspeed of a swallow", "unknown", 0.0),  # -> missing
]


class FakeClient:
    """Returns replies shaped like each prompt's declared output format."""

    def __init__(self, endpoint):
        self.calls = {"complete": 0, "complete_json": 0, "chat_json": 0}

    # stage 1
    def complete(self, prompt, max_tokens, temperature, top_p, logprobs=False, seed=None):
        self.calls["complete"] += 1
        question = prompt.split("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0].strip()
        gold, rate = next((g, r) for q, g, r in QUESTIONS if q == question)
        # Deterministic hit pattern so region assignment is predictable.
        hit = (self.calls["complete"] % 10) < int(rate * 10)
        return {
            "text": f"The answer is {gold}." if hit else "Something else entirely.",
            "avg_surprisal": 0.2 if rate > 0.9 else (1.0 if rate > 0.1 else 2.5),
        }

    # stages 2 and 3
    def complete_json(self, prompt, **kwargs):
        self.calls["complete_json"] += 1
        if "Search_Query" in prompt:
            return {"Search_Query": "reformulated retrieval query about radium"}
        # region assignment: mirror the mean accuracy the prompt carries
        acc = float(prompt.split("Mean Accuracy: ", 1)[1].split("\n", 1)[0])
        label = ("Knowledge Mastered" if acc >= 0.8
                 else "Knowledge Missing" if acc <= 0.2 else "Knowledge Confused")
        return {"Knowledge_Status": label, "Explanation": "Behaviour was consistent."}

    # stage 5
    def chat_json(self, messages, **kwargs):
        self.calls["chat_json"] += 1
        return {
            "qa_pairs": [
                {"question": "Who isolated radium in 1902?", "answer": "Marie Curie."},
                {"question": "", "answer": "dropped, no question"},
                {"question": "Leaks scaffolding?", "answer": "See the passage above."},
                {"question": "What element did she discover?", "answer": "Radium."},
            ]
        }


def main() -> int:
    llm.LLMClient = FakeClient
    for module in (s1_sample, s2_diagnose, s3_meta, s5_synthesize):
        module.LLMClient = FakeClient

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "data").mkdir()

        seed = root / "data" / "seed.jsonl"
        seed.write_text(
            "\n".join(json.dumps({"question": q, "answer": a}) for q, a, _ in QUESTIONS) + "\n"
        )

        cfg_path = root / "config.yaml"
        cfg_path.write_text(
            f"data_root: {root}/data\n"
            f"run_root: {root}/run\n"
            f"input_path: {seed}\n"
            "endpoint:\n  model: fake-model\n"
            "sampling:\n  n_samples: 10\n  workers: 2\n"
            "diagnosis:\n  workers: 2\n"
            "meta:\n  workers: 2\n"
            "synthesis:\n  workers: 2\n"
        )
        cfg = load_config(str(cfg_path))

        print("\n[stage 1: profiling]")
        s1_sample.run(cfg)
        profiles = load_jsonl(cfg.path("sample_path"))
        check("one record per question", len(profiles) == 3, f"got {len(profiles)}")
        check("K samples each", all(len(p["samples"]) == 10 for p in profiles))
        check("uncertainty recorded", all("mean_uncertainty" in p for p in profiles))
        check("accuracy recorded", all("mean_acc" in p for p in profiles))
        check("no legacy surprisal key", all("mean_surprisal" not in p for p in profiles))

        print("\n[stage 2: region assignment]")
        s2_diagnose.run(cfg)
        assigned = load_jsonl(cfg.path("diagnosis_path"))
        regions = {a["question"]: a["region"] for a in assigned}
        check("all assigned", len(assigned) == 3)
        check("high accuracy -> mastered", regions["who discovered radium"] == "mastered",
              f"got {regions['who discovered radium']}")
        check("mid accuracy -> confused", regions["when was the treaty signed"] == "confused",
              f"got {regions['when was the treaty signed']}")
        check("zero accuracy -> missing",
              regions["what is the airspeed of a swallow"] == "missing",
              f"got {regions['what is the airspeed of a swallow']}")
        check("paper label stored", all(a["knowledge_status"].startswith("Knowledge ")
                                        for a in assigned))
        check("no error_type emitted", all("error_type" not in a for a in assigned))
        check("source tagged", all(a["assignment_source"] == "meta_model" for a in assigned))

        print("\n[stage 3: cognitive tagging]")
        s3_meta.run(cfg)
        queried = load_jsonl(cfg.path("meta_path"))
        check("all have a query", all(q["knowledge_query"] for q in queried))
        check("query came from the model",
              all(q["query_source"] == "meta_model" for q in queried))
        check("no meta_recognition field", all("meta_recognition" not in q for q in queried))

        print("\n[stage 4: retrieval merge]")
        engine_out = root / "engine.json"
        engine_out.write_text(json.dumps([
            {"query": "reformulated retrieval query about radium",
             "passages": [{"contents": f"Passage {i} about radium."} for i in range(8)]}
        ]))
        s4_retrieve.run(cfg, results=str(engine_out), top_k=5)
        retrieved = load_jsonl(cfg.path("retrieval_path"))
        check("passages attached", all(len(r["retrieved_passages"]) == 5 for r in retrieved),
              str([len(r["retrieved_passages"]) for r in retrieved]))

        export = root / "queries.jsonl"
        s4_retrieve.run(cfg, export=str(export))
        check("export dedupes queries", len(load_jsonl(export)) == 1)

        print("\n[stage 5: synthesis]")
        s5_synthesize.run(cfg)
        synth = load_jsonl(cfg.path("synthesis_path"))
        check("all records synthesized", len(synth) == 3)
        check("bad pairs filtered", all(len(s["qa_pairs"]) == 2 for s in synth),
              str([len(s["qa_pairs"]) for s in synth]))
        check("scaffolding leak filtered",
              not any("passage" in json.dumps(s["qa_pairs"]).lower() for s in synth))
        check("pairs are question+answer only",
              all(set(qa) == {"question", "answer"} for s in synth for qa in s["qa_pairs"]))
        check("region_short stamped", all(s["region_short"] in {"MAS", "CON", "MIS"} for s in synth))

        print("\n[stage 6: build]")
        s6_build_sft.run(cfg, fmt="sft")
        rows = load_jsonl(cfg.path("sft_path"))
        check("flattened 3x2 rows", len(rows) == 6, f"got {len(rows)}")
        check("sft carries provenance", all(r["meta_query"] and r["region"] for r in rows))
        check("no think wrapper", not any("<think>" in r["answer"] for r in rows))
        check("all three regions present",
              {r["region"] for r in rows} == {"mastered", "confused", "missing"})

        s6_build_sft.run(cfg, fmt="qa")
        qa_rows = load_jsonl(cfg.path("sft_path"))
        check("qa format is 2 keys", all(set(r) == {"question", "answer"} for r in qa_rows))
        check("qa row count matches", len(qa_rows) == 6)

        print("\n[resume]")
        before = len(load_jsonl(cfg.path("sample_path")))
        s1_sample.run(cfg, resume=True)
        check("resume is a no-op when complete",
              len(load_jsonl(cfg.path("sample_path"))) == before)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("END-TO-END PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

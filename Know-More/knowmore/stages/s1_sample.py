"""Stage 1 -- behavioral profiling by stochastic decoding.

Sample K independent responses per query and record, for each, the correctness
and the internal uncertainty. Together these form the behavior profile

    B(q, a) = {(a_hat_k, u_k, l_k)}_{k=1..K}

where u is the sequence-level average negative log-likelihood and l is
correctness. Stage 2 maps this profile onto a knowledge region.

The sampling prompt is deliberately plain, so the profile reflects inherent
knowledge rather than instruction-following.

Input   a JSONL file of ``{"question": ..., "answer": ...}``
Output  one record per question with all samples plus ``mean_acc`` and
        ``mean_uncertainty``
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from ..config import Config
from ..io_utils import JSONLWriter, completed_keys, load_jsonl, require_input
from ..llm import LLMClient
from ..metrics import compute_accuracy, mean_or_nan
from ..prompts import SAMPLING_PROMPT

log = logging.getLogger(__name__)


def _sample_once(client: LLMClient, cfg: Config, question: str, seed: int) -> Dict[str, Any]:
    result = client.complete(
        prompt=SAMPLING_PROMPT.format(question=question),
        max_tokens=cfg.sampling.max_tokens,
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        logprobs=True,
        seed=seed,
    )
    # The client's avg_surprisal is the mean negative log-likelihood over
    # generated tokens, which is the uncertainty signal this pipeline uses.
    return {"response": result["text"], "uncertainty": result["avg_surprisal"]}


def run(cfg: Config, resume: bool = False) -> None:
    in_path = cfg.path("input_path")
    out_path = cfg.path("sample_path")
    require_input(in_path, stage="sample")

    dataset = load_jsonl(in_path)
    log.info("loaded %d questions from %s", len(dataset), in_path)

    done = completed_keys(out_path, "question") if resume else set()
    pending = [item for item in dataset if str(item.get("question", "")) not in done]
    if not pending:
        log.info("nothing to do; all %d questions already sampled", len(dataset))
        return

    client = LLMClient(cfg.endpoint)
    rng = np.random.default_rng()
    k = cfg.sampling.n_samples
    log.info("sampling K=%d responses for %d questions", k, len(pending))

    with JSONLWriter(out_path, resume=resume) as writer:
        # Questions are handled one at a time, but the K samples for a question
        # fan out across the pool. Keeping a question's samples together means a
        # record is only written once complete, which is what makes --resume
        # correct at question granularity.
        with ThreadPoolExecutor(max_workers=cfg.sampling.workers) as pool:
            for item in tqdm(pending, desc="Profiling", ncols=100):
                question = str(item.get("question", "")).strip()
                gold = str(item.get("answer", "")).strip()
                if not question:
                    continue

                futures = {
                    pool.submit(
                        _sample_once, client, cfg, question, int(rng.integers(1, 2**31 - 1))
                    ): i
                    for i in range(k)
                }

                samples: List[Dict[str, Any]] = []
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        sampled = future.result()
                    except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the run
                        log.warning("sample %d failed for %r: %s", idx, question[:60], exc)
                        continue
                    sampled["sample_id"] = idx
                    sampled["accuracy"] = compute_accuracy(sampled["response"], gold)
                    samples.append(sampled)

                if not samples:
                    log.warning("no usable samples for %r, skipping", question[:60])
                    continue

                samples.sort(key=lambda s: s["sample_id"])
                writer.write(
                    {
                        "question": question,
                        "gold_answer": gold,
                        "samples": samples,
                        "mean_acc": mean_or_nan([s["accuracy"] for s in samples]),
                        "mean_uncertainty": mean_or_nan([s["uncertainty"] for s in samples]),
                    }
                )

    log.info("%s now holds %d records", out_path, _count(out_path))


def _count(path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

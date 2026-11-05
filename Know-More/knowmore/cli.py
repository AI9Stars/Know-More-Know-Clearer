"""Command-line entry point.

    knowmore <stage> --config configs/example.yaml [--resume] [--set k=v ...]

Stages run in order and each reads the previous one's output, so a full run is:

    knowmore sample     --config c.yaml
    knowmore diagnose   --config c.yaml
    knowmore meta       --config c.yaml
    knowmore retrieve   --config c.yaml --export queries.jsonl
    #   ... run your retriever ...
    knowmore retrieve   --config c.yaml --results engine_out.json
    knowmore synthesize --config c.yaml
    knowmore build      --config c.yaml --format sft

``knowmore all`` runs everything except retrieval, which needs an external
engine and therefore a manual step.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List

from .config import load_config
from .stages import s1_sample, s2_diagnose, s3_meta, s4_retrieve, s5_synthesize, s6_build_sft

STAGE_ORDER = ["sample", "diagnose", "meta", "retrieve", "synthesize", "build"]


def _parse_overrides(pairs: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowmore",
        description="Know-More: meta-cognitive data construction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=STAGE_ORDER + ["all"], help="pipeline stage to run")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set sampling.n_samples=8 (repeatable)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip records already present in the stage's output file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="enable debug logging")

    group = parser.add_argument_group("retrieve")
    group.add_argument("--export", metavar="PATH", help="dump knowledge queries for your retriever")
    group.add_argument("--results", metavar="PATH", help="merge your retriever's output back in")
    group.add_argument("--top-k", type=int, default=None, help="keep at most K passages per query")

    group = parser.add_argument_group("build")
    group.add_argument(
        "--format",
        choices=s6_build_sft.FORMATS,
        default="sft",
        help="sft = <think>/<answer> targets; qa = plain question/answer for the RL stage",
    )

    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(message)s",
    )

    try:
        cfg = load_config(args.config, _parse_overrides(args.set))
    except (ValueError, FileNotFoundError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.stage == "all":
            s1_sample.run(cfg, resume=args.resume)
            s2_diagnose.run(cfg, resume=args.resume)
            s3_meta.run(cfg, resume=args.resume)
            logging.warning(
                "stopping before retrieval: it needs an external engine. "
                "Run `knowmore retrieve --export ...`, then `--results ...`, "
                "then `knowmore synthesize` and `knowmore build`."
            )
        elif args.stage == "sample":
            s1_sample.run(cfg, resume=args.resume)
        elif args.stage == "diagnose":
            s2_diagnose.run(cfg, resume=args.resume)
        elif args.stage == "meta":
            s3_meta.run(cfg, resume=args.resume)
        elif args.stage == "retrieve":
            s4_retrieve.run(cfg, results=args.results, export=args.export, top_k=args.top_k)
        elif args.stage == "synthesize":
            s5_synthesize.run(cfg, resume=args.resume)
        elif args.stage == "build":
            s6_build_sft.run(cfg, fmt=args.format)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; re-run with --resume to continue", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

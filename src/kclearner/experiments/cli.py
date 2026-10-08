"""Minimal corrected-EdNet command line.

Examples
--------
python -m kclearner.experiments.cli run --model irt --dry-run
python -m kclearner.experiments.cli run --model dkt_q --seed 42 --dry-run
python -m kclearner.experiments.cli run --model irt --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kclearner.experiments.config import formal_run_matrix, load_protocol, resolve_run
from kclearner.experiments.runner import dry_run, execute_formal, run_authorized, smoke_run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kclearner")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, default=None)
    run.add_argument("--model", required=True)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--dataset-dir", type=Path, default=None)
    run.add_argument("--runs-dir", type=Path, default=None)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    run.add_argument("--phase", choices=("train", "valid"), default="train")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    run.add_argument("--authorize-formal", action="store_true")
    args = parser.parse_args(argv)
    root = _repo_root()
    protocol_path = args.protocol or (root / "configs" / "ednet_corrected" / "protocol.json")
    dataset_dir = args.dataset_dir or (root / "generated" / "ednet_kt1_corrected_v1")
    runs_dir = args.runs_dir or (root / "runs")
    protocol = load_protocol(protocol_path)
    if args.device is not None:
        protocol["device"] = args.device
    config = resolve_run(protocol, args.model, args.seed)
    if args.authorize_formal:
        report = run_authorized(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            requested_device=protocol.get("device", "auto"),
        )
        print(json.dumps({"run_id": config.run_id, "status": "formal", "publication": True, "best_epoch": report["best_epoch"]}, sort_keys=True))
        return 0
    if args.smoke:
        report = smoke_run(config, dataset_dir=dataset_dir, runs_dir=runs_dir)
    elif args.dry_run:
        report = dry_run(config, dataset_dir=dataset_dir, runs_dir=runs_dir)
    else:
        execute_formal()
    print(json.dumps({"run_id": report["run_id"], "status": report.get("status", "smoke"), "publication": False}, sort_keys=True))
    return 0


def matrix_ids(protocol_path: str | Path | None = None) -> list[str]:
    path = Path(protocol_path) if protocol_path else _repo_root() / "configs" / "ednet_corrected" / "protocol.json"
    return [run.run_id for run in formal_run_matrix(load_protocol(path))]


if __name__ == "__main__":
    raise SystemExit(main())

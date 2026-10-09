"""Public dataset preparation and TRAIN/VALID runs.

Examples
--------
python -m kclearner.experiments.cli prepare --dataset ednet_kt1 --raw-dir <KT1> --questions <questions.csv>
python -m kclearner.experiments.cli prepare --dataset assistments2017 --raw-file <primary.csv>
python -m kclearner.experiments.cli run --dataset ednet_kt1 --model irt --dry-run
python -m kclearner.experiments.cli run --dataset assistments2017 --model ar_kt --dry-run
python -m kclearner.experiments.cli run-canonical --dataset-dir <canonical-dir>
python -m kclearner.experiments.cli run-external --dataset-dir <canonical-dir> --adapter module:Class --oov-state-semantics <id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kclearner.data.assistments_prepare import prepare_assistments2017
from kclearner.data.ednet_prepare import prepare_ednet_kt1
from kclearner.experiments.config import formal_run_matrix, load_protocol, resolve_run
from kclearner.experiments.runner import dry_run, execute_formal, run_authorized, smoke_run


def _repo_root() -> Path:
    """Directory that contains ``configs/``.

    An editable checkout keeps that directory four levels above this
    file. An installed wheel does not, so a checkout used as the
    working directory is accepted as well.
    """

    marker = Path("configs") / "ednet_corrected" / "protocol.json"
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / marker).is_file():
            return parent
    cwd = Path.cwd()
    if (cwd / marker).is_file():
        return cwd
    return start.parents[3]


def _add_prepare(sub: argparse._SubParsersAction) -> None:
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--dataset",
        required=True,
        choices=("ednet_kt1", "assistments2017"),
    )
    prepare.add_argument("--protocol", type=Path, default=None)
    prepare.add_argument("--raw-dir", type=Path, default=None)
    prepare.add_argument("--questions", type=Path, default=None)
    prepare.add_argument("--raw-file", type=Path, default=None)
    prepare.add_argument("--output-dir", type=Path, default=None)
    prepare.add_argument("--rebuild", action="store_true")


def _add_run(sub: argparse._SubParsersAction) -> None:
    run = sub.add_parser("run")
    run.add_argument(
        "--dataset",
        choices=("ednet_kt1", "assistments2017"),
        default="ednet_kt1",
    )
    run.add_argument("--protocol", type=Path, default=None)
    run.add_argument("--model", required=True)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--dataset-dir", type=Path, default=None)
    run.add_argument("--runs-dir", type=Path, default=None)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--authorize-formal", action="store_true")


def _add_run_canonical(sub: argparse._SubParsersAction) -> None:
    """Non-formal bundle execution for an external canonical directory.

    This parser has no model, protocol, freeze, or TEST arguments.
    ``run``, ``prepare``, and ``evaluate`` keep their reference dataset
    choices.
    """

    command = sub.add_parser("run-canonical")
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--runs-dir", type=Path, default=None)
    mode = command.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")


def _add_run_external(sub: argparse._SubParsersAction) -> None:
    """Non-formal TRAIN/VALID for one external execution adapter.

    There is no freeze or TEST switch. Reference ``run`` dataset choices
    stay unchanged.
    """

    command = sub.add_parser("run-external")
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--adapter", required=True)
    command.add_argument("--oov-state-semantics", required=True)
    command.add_argument("--runs-dir", type=Path, default=None)


def _add_evaluate(sub: argparse._SubParsersAction) -> None:
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--dataset",
        required=True,
        choices=("ednet_kt1", "assistments2017"),
    )
    evaluate.add_argument("--protocol", type=Path, default=None)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--seed", type=int, default=None)
    evaluate.add_argument("--freeze", type=Path, required=True)
    evaluate.add_argument("--dataset-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--enable-test-execution", action="store_true")


def _prepare_main(args: argparse.Namespace, root: Path, parser: argparse.ArgumentParser) -> int:
    if args.dataset == "ednet_kt1":
        if args.raw_file is not None:
            parser.error("ednet_kt1 prepare does not accept --raw-file")
        if args.raw_dir is None or args.questions is None:
            parser.error("ednet_kt1 prepare requires --raw-dir and --questions")
        protocol_path = args.protocol or (root / "configs" / "ednet_corrected" / "protocol.json")
        output_dir = args.output_dir or (root / "generated" / "ednet_kt1_corrected_v1")
        report = prepare_ednet_kt1(
            args.raw_dir,
            args.questions,
            output_dir,
            load_protocol(protocol_path, dataset="ednet_kt1"),
            rebuild=args.rebuild,
        )
        payload = {
            "status": "reused" if report.reused else "prepared",
            "dataset": "ednet_kt1",
            "output_dir": str(report.output_dir),
            "dataset_logical_hash": report.dataset_logical_hash,
            "pilot_sha256": report.pilot_sha256,
            "dev5000_sha256": report.dev5000_sha256,
            "item_ids_sha256": report.item_ids_sha256,
            "kc_ids_sha256": report.kc_ids_sha256,
            "row_counts": report.row_counts,
        }
    else:
        if args.raw_dir is not None or args.questions is not None:
            parser.error("assistments2017 prepare does not accept --raw-dir or --questions")
        if args.raw_file is None:
            parser.error("assistments2017 prepare requires --raw-file")
        protocol_path = args.protocol or (root / "configs" / "assistments2017" / "protocol.json")
        output_dir = args.output_dir or (root / "generated" / "assistments2017_main_v1")
        report = prepare_assistments2017(
            args.raw_file,
            output_dir,
            load_protocol(protocol_path, dataset="assistments2017"),
            rebuild=args.rebuild,
        )
        payload = {
            "status": "reused" if report.reused else "prepared",
            "dataset": "assistments2017",
            "output_dir": str(report.output_dir),
            "dataset_logical_hash": report.dataset_logical_hash,
            "csv_sha256": report.csv_sha256,
            "cohort_hash": report.cohort_hash,
            "item_ids_sha256": report.item_ids_sha256,
            "kc_ids_sha256": report.kc_ids_sha256,
            "row_counts": report.row_counts,
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _paths_for_run(args: argparse.Namespace, root: Path) -> tuple[Path, Path, Path]:
    if args.dataset == "assistments2017":
        protocol_path = args.protocol or (root / "configs" / "assistments2017" / "protocol.json")
        dataset_dir = args.dataset_dir or (root / "generated" / "assistments2017_main_v1")
    else:
        protocol_path = args.protocol or (root / "configs" / "ednet_corrected" / "protocol.json")
        dataset_dir = args.dataset_dir or (root / "generated" / "ednet_kt1_corrected_v1")
    runs_dir = args.runs_dir or (root / "runs")
    return protocol_path, dataset_dir, runs_dir


def _requested_device(args: argparse.Namespace, protocol: dict) -> str:
    if args.device is not None:
        return str(args.device)
    if args.authorize_formal:
        return str(protocol.get("device", "auto"))
    return "cpu"


def _run_main(args: argparse.Namespace, root: Path) -> int:
    protocol_path, dataset_dir, runs_dir = _paths_for_run(args, root)
    protocol = load_protocol(protocol_path, dataset=args.dataset)
    requested = _requested_device(args, protocol)
    config = resolve_run(
        protocol,
        args.model,
        args.seed,
        publication_protocol=bool(args.authorize_formal),
    )
    if config.model not in ("irt", "ar_kt") and requested == "cuda":
        from kclearner.experiments.neural_training import DeviceError, resolve_device

        try:
            resolve_device("cuda")
        except DeviceError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.authorize_formal:
        report = run_authorized(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            requested_device=requested,
        )
        print(json.dumps({
            "run_id": config.run_id,
            "status": "formal",
            "publication_protocol": True,
            "publication": True,
            "best_epoch": report["best_epoch"],
        }, sort_keys=True))
        return 0
    if args.smoke:
        report = smoke_run(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            requested_device=requested,
        )
    elif args.dry_run:
        report = dry_run(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            requested_device=requested,
        )
    else:
        execute_formal()
    print(json.dumps({
        "run_id": report["run_id"],
        "status": report.get("status", "smoke"),
        "publication_protocol": False,
        "publication": False,
    }, sort_keys=True))
    return 0


def _evaluate_main(args: argparse.Namespace, root: Path) -> int:
    from kclearner.experiments.evaluate import evaluate_prepared

    if args.dataset == "assistments2017":
        protocol_path = args.protocol or (root / "configs" / "assistments2017" / "protocol.json")
    else:
        protocol_path = args.protocol or (root / "configs" / "ednet_corrected" / "protocol.json")
    protocol = load_protocol(protocol_path, dataset=args.dataset)
    report = evaluate_prepared(
        args.freeze,
        args.dataset_dir,
        args.checkpoint,
        args.output_dir,
        enable_test=bool(args.enable_test_execution),
        protocol=protocol,
        model=args.model,
        seed=42 if args.seed is None else int(args.seed),
    )
    print(json.dumps({
        "phase": report["phase"],
        "role": report["role"],
        "freeze_id": report["freeze_id"],
        "metric_eligible_count": report["metric_eligible_count"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kclearner")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_prepare(sub)
    _add_run(sub)
    _add_run_canonical(sub)
    _add_run_external(sub)
    _add_evaluate(sub)
    args = parser.parse_args(argv)
    root = _repo_root()
    if args.command == "prepare":
        return _prepare_main(args, root, parser)
    if args.command == "evaluate":
        return _evaluate_main(args, root)
    if args.command == "run-canonical":
        return _run_canonical_main(args)
    if args.command == "run-external":
        return _run_external_main(args)
    return _run_main(args, root)


def _run_canonical_main(args: argparse.Namespace) -> int:
    from kclearner.experiments.canonical_run import execute_canonical_directory

    report = execute_canonical_directory(
        args.dataset_dir,
        runs_dir=args.runs_dir,
        dry_run=bool(args.dry_run),
        smoke=bool(args.smoke),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _run_external_main(args: argparse.Namespace) -> int:
    from kclearner.experiments.external_execution import execute_external_directory

    report = execute_external_directory(
        args.dataset_dir,
        args.adapter,
        {"oov_state_semantics_id": args.oov_state_semantics},
        runs_dir=args.runs_dir,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def matrix_ids(protocol_path: str | Path | None = None) -> list[str]:
    path = Path(protocol_path) if protocol_path else _repo_root() / "configs" / "ednet_corrected" / "protocol.json"
    return [run.run_id for run in formal_run_matrix(load_protocol(path))]


if __name__ == "__main__":
    raise SystemExit(main())

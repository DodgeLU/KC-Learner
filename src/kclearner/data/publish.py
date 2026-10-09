"""Same-filesystem publication of one prepared dataset directory.

Each caller receives its own staging directory. A destination is
replaced only when ``rebuild`` is true. A lost rename becomes
``reused`` only after the destination validates as the same logical
dataset.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable


class PublishError(RuntimeError):
    """The prepared directory could not be published safely."""


def make_staging_directory(destination: str | Path) -> Path:
    """Create an empty staging directory beside ``destination``.

    The directory is not the shared ``<destination>.incomplete`` path.
    ``tempfile.mkdtemp`` places it in ``destination``'s parent, so the
    later rename stays on the same filesystem.
    """

    target = Path(destination)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f"{target.name}.incomplete.", dir=parent)
    )
    if ".incomplete." not in staging.name:
        raise PublishError(f"staging path is not unique: {staging}")
    return staging


def discard_staging(staging: str | Path) -> None:
    """Remove this invocation's staging directory and nothing else."""

    path = Path(staging)
    if ".incomplete." not in path.name:
        raise PublishError(f"refusing to delete non-staging path {path}")
    if path.exists():
        shutil.rmtree(path)


def publish_prepared_directory(
    staging: str | Path,
    destination: str | Path,
    assess: Callable[[], str],
    *,
    rebuild: bool = False,
) -> str:
    """Publish ``staging`` or discard it.

    ``assess`` returns ``missing``, ``reuse``, or ``incompatible``.
    It is called immediately before publication and again if the rename
    fails. The return value is ``prepared`` or ``reused``.
    """

    staging_path = Path(staging)
    target = Path(destination)
    if not staging_path.is_dir():
        raise PublishError(f"staging directory does not exist: {staging_path}")
    if ".incomplete." not in staging_path.name:
        raise PublishError(f"refusing to publish non-staging path {staging_path}")
    decision = _decision(assess())
    if decision == "reuse" and not rebuild:
        discard_staging(staging_path)
        return "reused"
    if decision == "incompatible" and not rebuild:
        discard_staging(staging_path)
        raise PublishError(
            f"{target} exists and its logical identity does not match this "
            "preparation. The existing directory was left unchanged."
        )
    if decision == "missing":
        try:
            staging_path.rename(target)
        except OSError as exc:
            return _after_failed_rename(staging_path, target, assess, exc)
        return "prepared"
    return _replace(staging_path, target)


def _decision(value: str) -> str:
    if value not in ("missing", "reuse", "incompatible"):
        raise PublishError(f"unknown publication decision {value!r}")
    return value


def _after_failed_rename(
    staging: Path,
    destination: Path,
    assess: Callable[[], str],
    exc: OSError,
) -> str:
    try:
        decision = _decision(assess())
    except Exception as follow:
        discard_staging(staging)
        raise PublishError(
            f"could not publish {destination} and could not re-check it"
        ) from follow
    if decision == "reuse":
        discard_staging(staging)
        return "reused"
    discard_staging(staging)
    raise PublishError(
        f"could not publish {destination}; the destination is {decision}"
    ) from exc


def _replace(staging: Path, destination: Path) -> str:
    """Move a validated destination aside, then publish staging."""

    backup = destination.with_name(
        f"{destination.name}.previous.{uuid.uuid4().hex}"
    )
    if ".previous." not in backup.name:
        raise PublishError(f"refusing to use backup path {backup}")
    try:
        destination.rename(backup)
    except OSError as exc:
        discard_staging(staging)
        raise PublishError(
            f"could not move {destination} aside for --rebuild"
        ) from exc
    try:
        staging.rename(destination)
    except OSError as exc:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        discard_staging(staging)
        raise PublishError(
            f"--rebuild could not replace {destination}; the previous "
            "directory was restored when it was still available"
        ) from exc
    shutil.rmtree(backup)
    return "prepared"

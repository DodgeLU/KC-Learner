"""EdNet question-metadata KC parser.

This is the corrected encoding rule for a new data identity. It does not
read the legacy processed cohort and it does not reproduce
``ednet_kt1_preproc_v1``.

``questions.csv`` ``bundle_id`` is content identity. It is stored as
``content_bundle_id`` and is not the interaction bundle ``(user_id,
solving_id)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Mapping

CORRECTED_PREPROCESSING_ID = "ednet_question_tags_corrected_v1"
LEGACY_PREPROCESSING_ID = "ednet_kt1_preproc_v1"
_SENTINEL_TOKEN = "-1"


class EdNetTagParseError(ValueError):
    """A tags cell that is not a confirmed KC encoding."""

    def __init__(
        self,
        *,
        question_id: str | None,
        tags_raw: str,
        token: str,
        reason: str,
    ) -> None:
        self.question_id = question_id
        self.tags_raw = tags_raw
        self.token = token
        super().__init__(
            "invalid EdNet tag "
            f"question_id={question_id!r} raw_tags={tags_raw!r} "
            f"invalid_token={token!r}: {reason}"
        )


@dataclass(frozen=True)
class EdNetQuestionMetadata:
    """One question row. ``kc_ids`` length is not a row count."""

    question_id: str
    kc_ids: tuple[int, ...]
    tags_raw: str
    content_bundle_id: str | None
    correct_answer: str | None = None
    preprocessing_id: str = CORRECTED_PREPROCESSING_ID


def parse_ednet_tags(
    tags: str | None,
    *,
    question_id: str | None = None,
) -> tuple[int, ...]:
    """Parse an EdNet ``tags`` cell into canonical KC ids.

    Examples
    --------
    ``"5;2;182"`` → ``(5, 2, 182)``
    ``"-1"`` → ``()``
    ``""`` / ``None`` → ``()``
    ``"5;-1;2"`` → ``EdNetTagParseError``

    Only a whole cell whose stripped value is ``-1`` is the missing-KC
    sentinel. A ``-1`` token beside any other tag is undefined and
    raises. Non-integer tokens raise. Source order and repeated tokens
    are both kept. This is the legacy-compatible representation;
    deduplication is not applied.
    """

    if tags is None:
        return ()
    if not isinstance(tags, str):
        raise TypeError(f"tags must be str or None, got {type(tags).__name__}")
    raw = tags.strip()
    if raw == "" or raw == _SENTINEL_TOKEN:
        return ()

    ordered: list[int] = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if tok == "":
            continue
        if tok == _SENTINEL_TOKEN:
            raise EdNetTagParseError(
                question_id=question_id,
                tags_raw=tags,
                token=tok,
                reason=(
                    "only a standalone -1 cell is the missing-KC sentinel; "
                    "a -1 token mixed with other tags is undefined"
                ),
            )
        if not tok.isdigit():
            raise EdNetTagParseError(
                question_id=question_id,
                tags_raw=tags,
                token=tok,
                reason="tag token must be a non-negative integer",
            )
        ordered.append(int(tok))
    return tuple(ordered)


def load_ednet_question_metadata(
    source: str | Path | IO[str],
) -> tuple[EdNetQuestionMetadata, ...]:
    """Load ``question_id`` / ``tags`` rows from an EdNet ``questions.csv``.

    ``bundle_id``, when present, is returned as ``content_bundle_id``.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return _parse_question_rows(handle)
    return _parse_question_rows(source)


def _parse_question_rows(handle: Iterable[str]) -> tuple[EdNetQuestionMetadata, ...]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError("questions.csv has no header")
    fields = set(reader.fieldnames)
    if "question_id" not in fields or "tags" not in fields:
        raise ValueError(
            "questions.csv must contain question_id and tags columns, "
            f"got {reader.fieldnames}"
        )
    rows: list[EdNetQuestionMetadata] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        rows.append(_row_to_metadata(row, line_number, seen_ids))
    return tuple(rows)


def _row_to_metadata(
    row: Mapping[str, str | None],
    line_number: int,
    seen_ids: set[str],
) -> EdNetQuestionMetadata:
    question_id = (row.get("question_id") or "").strip()
    if question_id == "":
        raise ValueError(f"empty question_id at line {line_number}")
    if question_id in seen_ids:
        raise ValueError(f"duplicate question_id {question_id!r} at line {line_number}")
    seen_ids.add(question_id)
    tags_cell = row.get("tags")
    tags_raw = "" if tags_cell is None else str(tags_cell).strip()
    content_raw = row.get("bundle_id")
    content_bundle_id = None if content_raw is None else str(content_raw).strip()
    if content_bundle_id == "":
        content_bundle_id = None
    answer_raw = row.get("correct_answer")
    correct_answer = None if answer_raw is None else str(answer_raw).strip()
    if correct_answer == "":
        correct_answer = None
    return EdNetQuestionMetadata(
        question_id=question_id,
        kc_ids=parse_ednet_tags(tags_raw, question_id=question_id),
        tags_raw=tags_raw,
        content_bundle_id=content_bundle_id,
        correct_answer=correct_answer,
    )

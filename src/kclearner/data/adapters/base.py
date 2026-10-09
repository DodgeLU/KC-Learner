"""Shared adapter helpers.

Sources do not have to contain a bundle column. After an adapter
returns canonical interactions, every record has a non-empty
``bundle_id``.
"""

from __future__ import annotations


def ensure_bundle_id(source_bundle_id: str | None, interaction_id: str) -> str:
    """Use the source bundle, or ``interaction_id`` when the source has none."""

    if not isinstance(interaction_id, str) or interaction_id.strip() == "":
        raise ValueError("interaction_id is required to assign bundle_id")
    if source_bundle_id is None:
        return interaction_id
    if not isinstance(source_bundle_id, str):
        raise TypeError(
            "source_bundle_id must be str or None, "
            f"got {type(source_bundle_id).__name__}"
        )
    stripped = source_bundle_id.strip()
    if stripped == "":
        return interaction_id
    return stripped

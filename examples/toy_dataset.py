"""Print the synthetic dataset and its bundle pre-states."""

from kclearner import (
    DatasetValidator,
    make_toy_interactions,
    prediction_target_count,
    replay_bundle_pre_state,
)


def main() -> None:
    rows = make_toy_interactions()
    result = DatasetValidator().validate(rows)
    print(
        f"interactions={len(rows)} targets={prediction_target_count(rows)} ok={result.ok}"
    )
    for row in rows:
        print(
            f"{row.interaction_id} learner={row.learner_id} item={row.item_id} "
            f"kc_ids={row.kc_ids} bundle={row.bundle_id} split={row.split}"
        )
    print("pre-bundle update counts:")
    for step in replay_bundle_pre_state(rows):
        print(
            f"  {step.interaction_id} bundle={step.bundle_id} "
            f"pre={step.pre_bundle_update_count}"
        )


if __name__ == "__main__":
    main()

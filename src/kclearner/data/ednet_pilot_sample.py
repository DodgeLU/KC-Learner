"""Historical EdNet 50k pilot sampler, ``ednet_pilot50k_sampler_v1``.

The selection is the one recorded by
``project_c/outputs/ednet_select_pilot_subset.py``: lexicographically
sorted KT1 user ids, a seed-42 sample of 50,000, then the sample
sorted again. Seed 42 is not used by the later 70/10/20 split.

``random.Random.sample`` is not called. The selection loop and
``_randbelow`` are the CPython 3.11.7 implementations, fixed here so a
later change to ``sample`` cannot move the cohort. The generator is
``random.Random(seed).getrandbits``, which is CPython's MT19937.
On Python 3.11.7 this reproduces the frozen 50k hash. The historical
wrapper refuses any other result.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from kclearner.data.logical_identity import hash_ordered_ids

SAMPLER_ID = "ednet_pilot50k_sampler_v1"
HISTORICAL_SEED = 42
HISTORICAL_SAMPLE_SIZE = 50_000
HISTORICAL_SOURCE_COUNT = 784_309
HISTORICAL_SAMPLE_SHA256 = (
    "20b29ac2ebf6e8cca9e763f11a44d850c773cf04e27464c8c5b0e8e9c7566b96"
)
HISTORICAL_DEV5000_SIZE = 5_000
HISTORICAL_DEV5000_SHA256 = (
    "d9bb296cef94ba4ca0fa80ac8da98b7179b537dbce14bcb7555d021423bcaf9b"
)


class PilotSampleError(ValueError):
    """The population or the sample does not match the frozen pilot."""


def _randbelow(rng: random.Random, n: int) -> int:
    """CPython 3.11.7 ``Random._randbelow_with_getrandbits``."""

    if n <= 0:
        raise PilotSampleError(f"randbelow requires n > 0, got {n}")
    getrandbits = rng.getrandbits
    bits = n.bit_length()
    value = getrandbits(bits)
    while value >= n:
        value = getrandbits(bits)
    return value


def sample_without_replacement(
    population: Sequence[str],
    k: int,
    rng: random.Random,
) -> list[str]:
    """CPython 3.11.7 ``Random.sample`` for a sequence and ``counts=None``.

    ``k`` at most 5, or a population no larger than the historical
    set-size threshold, uses the pool swap. Larger draws use the
    selection set. This is the branch that drew the 50k pilot.
    """

    n = len(population)
    if isinstance(k, bool) or not isinstance(k, int):
        raise PilotSampleError("k must be int")
    if not 0 <= k <= n:
        raise PilotSampleError(f"sample size {k} is outside 0..{n}")
    result: list[str | None] = [None] * k
    setsize = 21
    if k > 5:
        setsize += 4 ** math.ceil(math.log(k * 3, 4))
    if n <= setsize:
        pool = list(population)
        for index in range(k):
            chosen = _randbelow(rng, n - index)
            result[index] = pool[chosen]
            pool[chosen] = pool[n - index - 1]
    else:
        selected: set[int] = set()
        for index in range(k):
            chosen = _randbelow(rng, n)
            while chosen in selected:
                chosen = _randbelow(rng, n)
            selected.add(chosen)
            result[index] = population[chosen]
    return [item for item in result if item is not None]


def _require_sorted_unique(user_ids: Sequence[str]) -> None:
    previous: str | None = None
    for user_id in user_ids:
        if not isinstance(user_id, str) or user_id == "":
            raise PilotSampleError("user ids must be non-empty strings")
        if previous is not None and user_id <= previous:
            raise PilotSampleError(
                "user ids must be strictly increasing in lexicographic order"
            )
        previous = user_id


def select_sorted_sample(
    sorted_user_ids: Sequence[str],
    *,
    k: int,
    seed: int,
) -> tuple[str, ...]:
    """Sample ``k`` ids and return them in lexicographic order.

    ``sorted_user_ids`` must already be strictly sorted. The helper
    does not sort the input, because an unsorted population would be
    a different protocol.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PilotSampleError("seed must be int")
    _require_sorted_unique(sorted_user_ids)
    rng = random.Random(seed)
    chosen = sample_without_replacement(sorted_user_ids, k, rng)
    return tuple(sorted(chosen))


def select_ednet_pilot50k(sorted_user_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the frozen 50k pilot, or raise if it is not that pilot.

    The population count and the sample hash are both gates. A mismatch
    does not select a replacement cohort.
    """

    if len(sorted_user_ids) != HISTORICAL_SOURCE_COUNT:
        raise PilotSampleError(
            f"EdNet pilot source has {len(sorted_user_ids)} users, "
            f"frozen population is {HISTORICAL_SOURCE_COUNT}"
        )
    selected = select_sorted_sample(
        sorted_user_ids,
        k=HISTORICAL_SAMPLE_SIZE,
        seed=HISTORICAL_SEED,
    )
    digest = hash_ordered_ids(selected)
    if digest != HISTORICAL_SAMPLE_SHA256:
        raise PilotSampleError(
            "ednet_pilot50k_sampler_v1 output does not match the frozen "
            f"50k hash {HISTORICAL_SAMPLE_SHA256}"
        )
    return selected


def dev5000_from_pilot(sorted_pilot_user_ids: Sequence[str]) -> tuple[str, ...]:
    """First 5,000 ids of the already sorted 50k pilot.

    Raises unless that prefix matches the frozen dev5000 hash.
    """

    if len(sorted_pilot_user_ids) != HISTORICAL_SAMPLE_SIZE:
        raise PilotSampleError(
            f"dev5000 expects the sorted 50k pilot, got {len(sorted_pilot_user_ids)}"
        )
    _require_sorted_unique(sorted_pilot_user_ids)
    selected = tuple(sorted_pilot_user_ids[:HISTORICAL_DEV5000_SIZE])
    if hash_ordered_ids(selected) != HISTORICAL_DEV5000_SHA256:
        raise PilotSampleError(
            "dev5000 prefix does not match the frozen cohort hash "
            f"{HISTORICAL_DEV5000_SHA256}"
        )
    return selected

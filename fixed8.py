from collections.abc import Sequence
from statistics import mean, pstdev


GROUP_SIZE = 8
ADVANTAGE_EPSILON = 1e-6


def group_advantages(rewards: Sequence[int]) -> list[float]:
    """Return one GRPO advantage per reward, in rollout order.

    Accept exactly eight binary rewards from one prompt. Use the population
    standard deviation plus 1e-6 and return zero advantages for a uniform group.
    Raise ValueError for a wrong group size or a nonbinary reward.
    """
    if len(rewards) != GROUP_SIZE:
        raise ValueError(f"Expected {GROUP_SIZE} rewards from one prompt")
    if any(reward not in (0, 1) for reward in rewards):
        raise ValueError("Rewards must be binary")

    center = mean(rewards)
    scale = pstdev(rewards) + ADVANTAGE_EPSILON
    return [(reward - center) / scale for reward in rewards]

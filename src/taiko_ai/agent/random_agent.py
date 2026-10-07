from __future__ import annotations

import random

from taiko_ai.environment.actions import ACTION_COUNT


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def select_action(self, observation: object) -> int:
        del observation
        return self._random.randrange(ACTION_COUNT)

from __future__ import annotations

from taiko_ai.environment.actions import Action
from taiko_ai.reward.reward import target_notes_per_second


def limit_note_density(
    action: int,
    time: float,
    last_note_time: float | None,
    difficulty: int,
) -> int:
    if action == Action.REST or last_note_time is None:
        return int(action)

    minimum_interval = 1.0 / target_notes_per_second(difficulty)
    if time - last_note_time < minimum_interval - 1e-9:
        return int(Action.REST)
    return int(action)

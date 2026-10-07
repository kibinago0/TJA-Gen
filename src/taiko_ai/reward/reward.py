from __future__ import annotations

from taiko_ai.environment.actions import Action


TARGET_NOTES_PER_MINUTE = (
    140.0,
    147.0,
    162.0,
    174.0,
    195.0,
    221.0,
    250.0,
    295.0,
    355.0,
    428.0,
)


def target_notes_per_second(difficulty: int) -> float:
    if not 1 <= difficulty <= len(TARGET_NOTES_PER_MINUTE):
        raise ValueError("difficulty must be between 1 and 10")
    return TARGET_NOTES_PER_MINUTE[difficulty - 1] / 60.0


def calculate_reward(
    action: int,
    onset: float,
    volume: float,
    on_beat: bool,
    interval: float,
    density: float,
    difficulty: int,
    previous_big: bool = False,
    continuity: float = 0.0,
    build_up: float = 0.0,
    phrase_length: float = 0.0,
) -> float:
    target_density = target_notes_per_second(difficulty)
    if action == Action.REST:
        return -2.0 * onset if onset >= 0.35 else 0.0

    reward = 6.0 * onset + 0.75 * volume
    reward += 0.5 if on_beat else 0.0
    if onset < 0.15 and volume < 0.25:
        reward -= 1.5
    if interval < 0.12:
        reward -= 2.0
    if action in (Action.BIG_DON, Action.BIG_KA):
        if onset >= 0.7 and volume >= 0.7:
            reward += 2.0
        else:
            reward -= 2.0
        if previous_big:
            reward -= 2.5
    elif action == Action.ROLL_START:
        if continuity >= 0.4 and phrase_length >= 0.5:
            reward += 5.0
        else:
            reward -= 5.0
    elif action == Action.BALLOON_START:
        if build_up >= 0.15 and volume >= 0.5:
            reward += 5.0
        else:
            reward -= 5.0

    density_before = abs(target_density - density)
    density_after = abs(target_density - (density + 1.0))
    reward += 0.75 * (density_before - density_after)
    return reward

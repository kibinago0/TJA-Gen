from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from taiko_ai.environment.actions import ACTION_COUNT, Action

NOTE_HISTORY_LENGTH = 8
NOTE_HISTORY_CATEGORIES = 3
BASE_OBSERVATION_SIZE = 9
SPECIAL_FEATURE_COUNT = 3
NOTE_HISTORY_START = BASE_OBSERVATION_SIZE
SPECIAL_FEATURE_START = NOTE_HISTORY_START + NOTE_HISTORY_LENGTH * NOTE_HISTORY_CATEGORIES
OBSERVATION_SIZE = (
    BASE_OBSERVATION_SIZE
    + NOTE_HISTORY_LENGTH * NOTE_HISTORY_CATEGORIES
    + SPECIAL_FEATURE_COUNT
)


def make_observation(
    *,
    time: float,
    duration: float,
    bpm: float,
    onset: float,
    volume: float,
    previous_action: int,
    interval: float,
    density: float,
    difficulty: int,
    note_history: Sequence[int] = (),
    continuity: float = 0.0,
    build_up: float = 0.0,
    phrase_length: float = 0.0,
) -> np.ndarray:
    beat_phase = (time * bpm / 60.0) % 1.0
    recent_notes = list(note_history)[-NOTE_HISTORY_LENGTH:]
    if any(note not in (Action.DON, Action.KA) for note in recent_notes):
        raise ValueError("note_history can contain only DON and KA notes")
    padded_notes: list[int | None] = (
        [None] * (NOTE_HISTORY_LENGTH - len(recent_notes)) + recent_notes
    )
    encoded_history = np.zeros(
        (NOTE_HISTORY_LENGTH, NOTE_HISTORY_CATEGORIES), dtype=np.float32
    )
    for index, note in enumerate(padded_notes):
        category = 0 if note is None else int(note)
        encoded_history[index, category] = 1.0

    return np.asarray(
        np.concatenate(
            (
                np.asarray(
                    [
                        onset,
                        volume,
                        np.sin(2.0 * np.pi * beat_phase),
                        np.cos(2.0 * np.pi * beat_phase),
                        previous_action / (ACTION_COUNT - 1),
                        min(interval, 2.0) / 2.0,
                        min(density, 20.0) / 20.0,
                        difficulty / 10.0,
                        time / max(duration, 1e-6),
                    ],
                    dtype=np.float32,
                ),
                encoded_history.ravel(),
                np.asarray(
                    [
                        np.clip(continuity, 0.0, 1.0),
                        np.clip(build_up, -1.0, 1.0),
                        np.clip(phrase_length, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            )
        ),
        dtype=np.float32,
    )

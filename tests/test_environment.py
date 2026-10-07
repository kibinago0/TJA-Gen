import numpy as np

from taiko_ai.audio.analyzer import AudioFeatures
from taiko_ai.environment.actions import Action
from taiko_ai.environment.difficulty import limit_note_density
from taiko_ai.environment.state import (
    BASE_OBSERVATION_SIZE,
    NOTE_HISTORY_CATEGORIES,
    NOTE_HISTORY_LENGTH,
    NOTE_HISTORY_START,
    SPECIAL_FEATURE_START,
    OBSERVATION_SIZE,
    make_observation,
)
from taiko_ai.environment.taiko_env import TaikoEnv
from taiko_ai.reward.reward import calculate_reward, target_notes_per_second


def make_features() -> AudioFeatures:
    times = np.arange(0.0, 0.51, 0.01, dtype=np.float32)
    onset = np.zeros_like(times)
    onset[::12] = 1.0
    return AudioFeatures(
        duration=0.5,
        bpm=120.0,
        times=times,
        onset_strength=onset,
        volume=np.ones_like(times),
        onsets=times[onset > 0],
        beats=np.asarray([0.0, 0.5], dtype=np.float32),
    )


def test_environment_reaches_done_and_records_note_times() -> None:
    env = TaikoEnv(make_features(), difficulty=3)
    observation = env.reset()
    assert observation.shape == (OBSERVATION_SIZE,)

    results = []
    while True:
        observation, reward, done, info = env.step(Action.DON)
        results.append((reward, info))
        if done:
            break

    assert done
    assert len(env.note_times) == 4
    assert results[0][1]["time"] == 0.0


def test_observation_encodes_previous_eight_notes_oldest_to_newest() -> None:
    observation = make_observation(
        time=0.0,
        duration=1.0,
        bpm=120.0,
        onset=0.0,
        volume=0.0,
        previous_action=Action.KA,
        interval=0.25,
        density=2.0,
        difficulty=5,
        note_history=[
            Action.DON,
            Action.KA,
            Action.DON,
            Action.KA,
            Action.DON,
            Action.KA,
            Action.DON,
            Action.KA,
            Action.DON,
        ],
    )
    history = observation[NOTE_HISTORY_START:SPECIAL_FEATURE_START].reshape(
        NOTE_HISTORY_LENGTH, NOTE_HISTORY_CATEGORIES
    )

    assert observation.shape == (OBSERVATION_SIZE,)
    assert history.argmax(axis=1).tolist() == [
        Action.KA,
        Action.DON,
        Action.KA,
        Action.DON,
        Action.KA,
        Action.DON,
        Action.KA,
        Action.DON,
    ]
    assert np.all(history.sum(axis=1) == 1.0)


def test_environment_resets_and_updates_recent_note_history() -> None:
    env = TaikoEnv(make_features(), difficulty=3)
    observation = env.reset()
    env.step(Action.DON)
    observation, _, _, _ = env.step(Action.KA)
    history = observation[NOTE_HISTORY_START:SPECIAL_FEATURE_START].reshape(
        NOTE_HISTORY_LENGTH, NOTE_HISTORY_CATEGORIES
    )

    assert history.argmax(axis=1)[-2:].tolist() == [Action.DON, Action.KA]
    reset_observation = env.reset()
    reset_history = reset_observation[
        NOTE_HISTORY_START:SPECIAL_FEATURE_START
    ].reshape(
        NOTE_HISTORY_LENGTH, NOTE_HISTORY_CATEGORIES
    )
    assert reset_history.argmax(axis=1).tolist() == [0] * NOTE_HISTORY_LENGTH


def test_big_note_actions_share_hand_history_and_track_repetition() -> None:
    env = TaikoEnv(make_features(), difficulty=3)
    env.reset()
    _, _, _, first_info = env.step(Action.BIG_DON)
    observation, _, _, _ = env.step(Action.BIG_KA)
    history = observation[NOTE_HISTORY_START:SPECIAL_FEATURE_START].reshape(
        NOTE_HISTORY_LENGTH, NOTE_HISTORY_CATEGORIES
    )

    assert first_info["action"] == Action.BIG_DON
    assert history.argmax(axis=1)[-2:].tolist() == [Action.DON, Action.KA]
    assert env.previous_note_was_big

    env.step(Action.REST)
    assert not env.previous_note_was_big


def test_reward_prefers_notes_at_onsets_over_rest() -> None:
    note_reward = calculate_reward(Action.DON, 1.0, 1.0, True, 1.0, 0.0, 5)
    rest_reward = calculate_reward(Action.REST, 1.0, 1.0, True, 1.0, 0.0, 5)

    assert note_reward > rest_reward


def test_big_notes_reward_strong_accents_and_discourage_weak_or_repeated_notes() -> None:
    strong_big = calculate_reward(
        Action.BIG_DON, 0.9, 0.9, True, 1.0, 0.0, 5
    )
    regular = calculate_reward(Action.DON, 0.9, 0.9, True, 1.0, 0.0, 5)
    weak_big = calculate_reward(
        Action.BIG_DON, 0.2, 0.2, False, 1.0, 0.0, 5
    )
    repeated_big = calculate_reward(
        Action.BIG_KA, 0.9, 0.9, True, 1.0, 0.0, 5, previous_big=True
    )

    assert strong_big > regular
    assert weak_big < regular
    assert repeated_big < strong_big


def test_target_note_density_increases_with_difficulty() -> None:
    targets = [target_notes_per_second(level) for level in range(1, 11)]

    assert targets == sorted(targets)
    assert targets[0] < targets[4] < targets[-1]


def test_density_reward_prefers_difficulty_specific_note_counts() -> None:
    low_level = calculate_reward(Action.DON, 0.5, 0.5, False, 1.0, 3.0, 1)
    high_level = calculate_reward(Action.DON, 0.5, 0.5, False, 1.0, 3.0, 10)

    assert high_level > low_level


def test_difficulty_density_limiter_enforces_level_specific_note_intervals() -> None:
    low_level_interval = 1.0 / target_notes_per_second(1)
    high_level_interval = 1.0 / target_notes_per_second(10)

    assert (
        limit_note_density(Action.DON, 0.2, 0.0, difficulty=1)
        == Action.REST
    )
    assert (
        limit_note_density(Action.KA, 0.2, 0.0, difficulty=10)
        == Action.KA
    )
    assert (
        limit_note_density(
            Action.DON,
            low_level_interval,
            0.0,
            difficulty=1,
        )
        == Action.DON
    )
    assert (
        limit_note_density(Action.REST, 0.01, 0.0, difficulty=1)
        == Action.REST
    )


def test_special_note_state_contains_audio_phrase_features() -> None:
    env = TaikoEnv(make_features(), difficulty=3)
    observation = env.reset()
    special_features = observation[SPECIAL_FEATURE_START:]

    assert special_features.shape == (3,)
    assert np.all(np.isfinite(special_features))
    assert 0.0 <= special_features[0] <= 1.0
    assert -1.0 <= special_features[1] <= 1.0
    assert 0.0 <= special_features[2] <= 1.0


def test_active_roll_discourages_overlapping_notes() -> None:
    env = TaikoEnv(make_features(), difficulty=3)
    env.reset()
    _, _, _, roll_info = env.step(Action.ROLL_START)

    _, overlap_reward, _, _ = env.step(Action.DON)
    _, rest_reward, _, _ = env.step(Action.REST)

    assert roll_info["special_duration"] >= 0.5
    assert overlap_reward < rest_reward

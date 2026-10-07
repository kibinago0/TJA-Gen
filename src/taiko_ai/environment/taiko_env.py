from __future__ import annotations

from collections import deque

import numpy as np

from taiko_ai.audio.analyzer import AudioFeatures
from taiko_ai.chart.patterns import HumanPatternModel
from taiko_ai.environment.actions import Action, ACTION_COUNT
from taiko_ai.environment.state import NOTE_HISTORY_LENGTH, make_observation
from taiko_ai.reward.reward import calculate_reward


class TaikoEnv:
    def __init__(
        self,
        features: AudioFeatures,
        difficulty: int = 5,
        subdivisions: int = 4,
        pattern_model: HumanPatternModel | None = None,
    ) -> None:
        if features.duration <= 0:
            raise ValueError("audio duration must be positive")
        if features.bpm <= 0:
            raise ValueError("BPM must be positive")
        if not 1 <= difficulty <= 10:
            raise ValueError("difficulty must be between 1 and 10")
        if subdivisions <= 0:
            raise ValueError("subdivisions must be positive")

        self.features = features
        self.difficulty = difficulty
        self.subdivisions = subdivisions
        self.pattern_model = pattern_model
        self.step_duration = 60.0 / features.bpm / subdivisions
        self.max_steps = max(1, int(np.ceil(features.duration / self.step_duration)))
        self.index = 0
        self.previous_action = int(Action.REST)
        self.last_note_time: float | None = None
        self.previous_note_was_big = False
        self.note_times: list[float] = []
        self.recent_notes: deque[float] = deque()
        self.note_history: deque[int] = deque(maxlen=NOTE_HISTORY_LENGTH)
        self.special_until = 0.0

    def reset(self) -> np.ndarray:
        self.index = 0
        self.previous_action = int(Action.REST)
        self.last_note_time = None
        self.previous_note_was_big = False
        self.note_times = []
        self.recent_notes.clear()
        self.note_history.clear()
        self.special_until = 0.0
        return self._observation()

    def _sample(self, values: np.ndarray, time: float) -> float:
        times = self.features.times
        if len(times) == 0:
            return 0.0
        left = int(np.searchsorted(times, time - 0.04, side="left"))
        right = int(np.searchsorted(times, time + 0.04, side="right"))
        right = min(max(right, left + 1), len(values))
        left = min(left, len(values) - 1)
        return float(np.max(values[left:right]))

    def _special_audio_features(self, time: float) -> tuple[float, float, float]:
        times = self.features.times
        if len(times) == 0:
            return 0.0, 0.0, 0.0

        start = int(np.searchsorted(times, time, side="left"))
        end = min(
            int(np.searchsorted(times, time + 2.0, side="right")),
            len(times),
        )
        future_volume = self.features.volume[start:end]
        future_onset = self.features.onset_strength[start:end]
        if len(future_volume) == 0:
            return 0.0, 0.0, 0.0

        active = future_volume >= 0.2
        continuity = float(np.mean(active))
        half = max(1, len(future_volume) // 2)
        previous_volume = self._sample(
            self.features.volume,
            max(0.0, time - 1.0),
        )
        build_up = float(
            np.clip(
                np.mean(future_volume[:half]) - previous_volume,
                -1.0,
                1.0,
            )
        )
        frame_period = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        active_duration = (
            int(np.argmax(~active)) * frame_period
            if not np.all(active)
            else len(active) * frame_period
        )
        phrase_length = min(1.0, active_duration / 2.0)
        if np.max(future_onset, initial=0.0) < 0.1:
            continuity *= 0.5
        return continuity, build_up, phrase_length

    def _observation(self) -> np.ndarray:
        time = min(self.index * self.step_duration, self.features.duration)
        while self.recent_notes and self.recent_notes[0] < time - 1.0:
            self.recent_notes.popleft()
        interval = (
            time - self.last_note_time if self.last_note_time is not None else 2.0
        )
        continuity, build_up, phrase_length = self._special_audio_features(time)
        return make_observation(
            time=time,
            duration=self.features.duration,
            bpm=self.features.bpm,
            onset=self._sample(self.features.onset_strength, time),
            volume=self._sample(self.features.volume, time),
            previous_action=self.previous_action,
            interval=interval,
            density=float(len(self.recent_notes)),
            difficulty=self.difficulty,
            note_history=self.note_history,
            continuity=continuity,
            build_up=build_up,
            phrase_length=phrase_length,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | int]]:
        if self.index >= self.max_steps:
            raise RuntimeError("episode is finished; call reset() before step()")
        if int(action) not in range(ACTION_COUNT):
            raise ValueError(f"action must be between 0 and {ACTION_COUNT - 1}")

        time = self.index * self.step_duration
        onset = self._sample(self.features.onset_strength, time)
        volume = self._sample(self.features.volume, time)
        while self.recent_notes and self.recent_notes[0] < time - 1.0:
            self.recent_notes.popleft()
        interval = (
            time - self.last_note_time if self.last_note_time is not None else 2.0
        )
        beat_fraction = (time * self.features.bpm / 60.0) % 1.0
        continuity, build_up, phrase_length = self._special_audio_features(time)
        special_active = time < self.special_until
        reward = calculate_reward(
            int(action),
            onset,
            volume,
            min(beat_fraction, 1.0 - beat_fraction) < 0.05,
            interval,
            float(len(self.recent_notes)),
            self.difficulty,
            previous_big=self.previous_note_was_big,
            continuity=continuity,
            build_up=build_up,
            phrase_length=phrase_length,
        )
        if special_active and action != Action.REST:
            reward -= 10.0
        if action != Action.REST:
            if self.pattern_model is not None:
                reward += self.pattern_model.score(self.note_history, int(action))
            if action in (Action.DON, Action.BIG_DON):
                self.note_history.append(int(Action.DON))
            elif action in (Action.KA, Action.BIG_KA):
                self.note_history.append(int(Action.KA))
            else:
                self.note_history.clear()
            self.last_note_time = time
            self.note_times.append(time)
            self.recent_notes.append(time)
            special_duration = 0.0
            if action == Action.ROLL_START:
                special_duration = max(0.5, min(2.0, phrase_length * 2.0))
            elif action == Action.BALLOON_START:
                special_duration = max(0.5, min(2.0, 0.5 + build_up * 1.5))
            if special_duration > 0:
                self.special_until = time + special_duration
            elif not special_active:
                self.special_until = time
            self.previous_note_was_big = action in (
                Action.BIG_DON,
                Action.BIG_KA,
            )
        else:
            self.previous_note_was_big = False
        self.previous_action = int(action)
        self.index += 1
        done = self.index >= self.max_steps
        return self._observation(), reward, done, {
            "time": time,
            "onset": onset,
            "volume": volume,
            "action": int(action),
            "continuity": continuity,
            "build_up": build_up,
            "phrase_length": phrase_length,
            "special_duration": max(0.0, self.special_until - time)
            if action in (Action.ROLL_START, Action.BALLOON_START)
            else 0.0,
        }

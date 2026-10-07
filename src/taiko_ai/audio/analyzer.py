from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioFeatures:
    duration: float
    bpm: float
    times: np.ndarray
    onset_strength: np.ndarray
    volume: np.ndarray
    onsets: np.ndarray
    beats: np.ndarray

    def with_bpm(self, bpm: float) -> AudioFeatures:
        if bpm <= 0:
            raise ValueError("BPM must be positive")
        return AudioFeatures(
            duration=self.duration,
            bpm=float(bpm),
            times=self.times,
            onset_strength=self.onset_strength,
            volume=self.volume,
            onsets=self.onsets,
            beats=np.arange(0.0, self.duration, 60.0 / bpm, dtype=np.float32),
        )

    def limited(self, duration: float) -> AudioFeatures:
        if duration <= 0:
            raise ValueError("duration must be positive")
        limit = min(float(duration), self.duration)
        keep = self.times < limit
        return AudioFeatures(
            duration=limit,
            bpm=self.bpm,
            times=self.times[keep],
            onset_strength=self.onset_strength[keep],
            volume=self.volume[keep],
            onsets=self.onsets[self.onsets < limit],
            beats=self.beats[self.beats < limit],
        )


def _estimate_bpm(onset_strength: np.ndarray, frame_period: float) -> float:
    signal = onset_strength.astype(np.float64, copy=False)
    signal = signal - signal.mean()
    if signal.size < 2 or not np.any(signal):
        return 120.0

    fft_size = 1 << (2 * signal.size - 1).bit_length()
    spectrum = np.fft.rfft(signal, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)
    autocorrelation = autocorrelation[: signal.size]
    min_lag = max(1, int(round(60.0 / 200.0 / frame_period)))
    max_lag = min(signal.size - 1, int(round(60.0 / 60.0 / frame_period)))
    if max_lag < min_lag:
        return 120.0

    lags = np.arange(min_lag, max_lag + 1)
    scores = autocorrelation[lags] / np.sqrt(lags)
    best_lag = int(lags[int(np.argmax(scores))])
    return float(np.clip(60.0 / (best_lag * frame_period), 60.0, 200.0))


def analyze_audio(path: str | Path, max_duration: float | None = None) -> AudioFeatures:
    source = Path(path)
    if max_duration is not None and max_duration <= 0:
        raise ValueError("max_duration must be positive")

    with sf.SoundFile(source) as audio_file:
        sample_rate = int(audio_file.samplerate)
        frame_limit = (
            int(max_duration * sample_rate)
            if max_duration is not None
            else -1
        )
        samples = audio_file.read(frames=frame_limit, dtype="float32", always_2d=True)

    if samples.size == 0:
        raise ValueError(f"Audio file contains no samples: {source}")
    mono = samples.mean(axis=1, dtype=np.float32)
    duration = mono.size / sample_rate
    window_size = max(128, int(round(sample_rate * 0.046)))
    hop_size = max(1, int(round(sample_rate * 0.01)))
    frame_count = max(1, 1 + int(np.ceil(max(0, mono.size - window_size) / hop_size)))
    padded_size = (frame_count - 1) * hop_size + window_size
    if mono.size < padded_size:
        mono = np.pad(mono, (0, padded_size - mono.size))

    squared = np.square(mono, dtype=np.float64)
    cumulative_energy = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    rms = np.sqrt(
        np.maximum(
            0.0,
            (cumulative_energy[starts + window_size] - cumulative_energy[starts])
            / window_size,
        )
    )
    times = (starts + window_size / 2) / sample_rate
    frame_period = hop_size / sample_rate
    energy_change = np.maximum(0.0, np.diff(np.log1p(rms * 100.0), prepend=0.0))
    onset_scale = float(np.percentile(energy_change, 95))
    onset_strength = (
        np.clip(energy_change / onset_scale, 0.0, 1.0)
        if onset_scale > 1e-8
        else np.zeros_like(energy_change)
    )
    volume_scale = float(np.percentile(rms, 95))
    volume = (
        np.clip(rms / volume_scale, 0.0, 1.0)
        if volume_scale > 1e-8
        else np.zeros_like(rms)
    )
    bpm = _estimate_bpm(onset_strength, frame_period)

    peaks = np.flatnonzero(
        (onset_strength >= 0.35)
        & (onset_strength >= np.roll(onset_strength, 1))
        & (onset_strength > np.roll(onset_strength, -1))
    )
    peaks = peaks[peaks > 0]
    if peaks.size > 1:
        separated = [int(peaks[0])]
        for peak in peaks[1:]:
            if times[peak] - times[separated[-1]] >= 0.08:
                separated.append(int(peak))
            elif onset_strength[peak] > onset_strength[separated[-1]]:
                separated[-1] = int(peak)
        peaks = np.asarray(separated, dtype=np.int64)

    onsets = times[peaks]
    beat_period = 60.0 / bpm
    beats = np.arange(0.0, duration, beat_period, dtype=np.float64)
    return AudioFeatures(
        duration=float(duration),
        bpm=bpm,
        times=times.astype(np.float32),
        onset_strength=onset_strength.astype(np.float32),
        volume=volume.astype(np.float32),
        onsets=onsets.astype(np.float32),
        beats=beats.astype(np.float32),
    )

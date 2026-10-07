import numpy as np
import soundfile as sf

from taiko_ai.audio.analyzer import AudioFeatures, analyze_audio


def test_analyzer_extracts_audio_features_from_wav(tmp_path) -> None:
    sample_rate = 22050
    audio = np.zeros(sample_rate, dtype=np.float32)
    for onset in np.arange(0.0, 1.0, 0.5):
        index = int(onset * sample_rate)
        audio[index : index + 300] = np.hanning(600)[::2]
    path = tmp_path / "sample.wav"
    sf.write(path, audio, sample_rate)

    features = analyze_audio(path)

    assert 0 < features.duration <= 1.0
    assert 60 <= features.bpm <= 200
    assert features.times.size > 0
    assert features.onset_strength.max() <= 1.0
    assert features.volume.max() <= 1.0


def test_analyzer_estimates_click_track_tempo(tmp_path) -> None:
    sample_rate = 22050
    audio = np.zeros(sample_rate * 12, dtype=np.float32)
    pulse = np.hanning(800)[::2]
    for time in np.arange(0.0, 12.0, 0.5):
        start = int(time * sample_rate)
        audio[start : start + pulse.size] = pulse
    path = tmp_path / "clicks.wav"
    sf.write(path, audio, sample_rate)

    features = analyze_audio(path)

    assert abs(features.bpm - 120.0) < 3.0


def test_audio_features_accept_tja_tempo_reference() -> None:
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    features = AudioFeatures(
        duration=1.0,
        bpm=90.0,
        times=times,
        onset_strength=np.zeros_like(times),
        volume=np.zeros_like(times),
        onsets=np.empty(0, dtype=np.float32),
        beats=np.asarray([0.0], dtype=np.float32),
    )

    referenced = features.with_bpm(120.0)

    assert features.bpm == 90.0
    assert referenced.bpm == 120.0
    np.testing.assert_allclose(referenced.beats, [0.0, 0.5])

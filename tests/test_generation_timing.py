import argparse
from pathlib import Path

import numpy as np
import pytest

from generate import (
    _resolve_timing,
    _tja_note_value,
    _validate_args,
    _wave_reference,
    _write_tja_atomically,
    build_parser,
)
from taiko_ai.environment.actions import Action
from taiko_ai.audio.analyzer import AudioFeatures


def make_features() -> AudioFeatures:
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float32)
    return AudioFeatures(
        duration=1.0,
        bpm=120.0,
        times=times,
        onset_strength=np.zeros_like(times),
        volume=np.zeros_like(times),
        onsets=np.empty(0, dtype=np.float32),
        beats=np.asarray([0.0, 0.5], dtype=np.float32),
    )


def test_timing_defaults_to_analyzed_bpm_and_zero_offset() -> None:
    features = make_features()

    resolved, offset = _resolve_timing(features, bpm=None, offset=None)

    assert resolved is features
    assert resolved.bpm == 120.0
    assert offset == 0.0


def test_manual_bpm_and_offset_override_automatic_values() -> None:
    resolved, offset = _resolve_timing(make_features(), bpm=150.0, offset=-1.25)

    assert resolved.bpm == 150.0
    np.testing.assert_allclose(resolved.beats, [0.0, 0.4, 0.8])
    assert offset == -1.25


def test_timing_values_can_be_overridden_independently() -> None:
    parser = build_parser()

    manual_bpm = parser.parse_args(["song.ogg", "--bpm", "150"])
    manual_offset = parser.parse_args(["song.ogg", "--offset", "-1.25"])

    assert manual_bpm.bpm == 150.0
    assert manual_bpm.offset is None
    assert manual_offset.bpm is None
    assert manual_offset.offset == -1.25


def test_cli_accepts_case_insensitive_difficulty_names() -> None:
    arguments = build_parser().parse_args(["song.wav", "--difficulty", "oni"])

    assert arguments.difficulty == "Oni"


def test_generation_pattern_seed_is_reproducible_by_default_and_overridable() -> None:
    parser = build_parser()

    assert parser.parse_args(["song.wav"]).seed == 7
    assert parser.parse_args(["song.wav", "--seed", "42"]).seed == 42


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--bpm", "nan"),
        ("--bpm", "0"),
        ("--offset", "inf"),
        ("--max-seconds", "nan"),
        ("--maximum-density", "-1"),
    ],
)
def test_cli_rejects_non_finite_or_non_positive_generation_values(option, value) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["song.wav", option, value])


def test_manual_bpm_must_be_positive() -> None:
    with pytest.raises(ValueError, match="BPM must be positive"):
        _resolve_timing(make_features(), bpm=0.0, offset=None)


def test_wave_reference_is_filename_only() -> None:
    audio = Path(r"C:\audio\Chozetsu Dynamic! (2023 Master).mp3")

    assert _wave_reference(audio) == "Chozetsu Dynamic! (2023 Master).mp3"


@pytest.mark.parametrize(
    ("action", "note_value"),
    [
        (Action.REST, None),
        (Action.DON, 1),
        (Action.KA, 2),
        (Action.BIG_DON, 3),
        (Action.BIG_KA, 4),
        (Action.ROLL_START, 5),
        (Action.BALLOON_START, 7),
    ],
)
def test_generated_actions_map_to_tja_note_values(action, note_value) -> None:
    assert _tja_note_value(action) == note_value


def test_generation_refuses_to_overwrite_audio_or_model(tmp_path) -> None:
    audio = tmp_path / "song.ogg"
    model = tmp_path / "model.tja"
    audio.touch()
    model.touch()
    common = {
        "audio": audio,
        "model": model,
        "level": 5,
        "bpm": None,
        "offset": None,
        "max_seconds": None,
        "maximum_density": None,
    }

    with pytest.raises(ValueError, match="must not overwrite"):
        _validate_args(
            argparse.Namespace(**common, output=model)
        )


def test_tja_write_replaces_existing_file_atomically(tmp_path) -> None:
    output = tmp_path / "nested" / "song.tja"

    _write_tja_atomically(output, "TITLE:First\n")
    _write_tja_atomically(output, "TITLE:Second\n")

    assert output.read_text(encoding="utf-8") == "TITLE:Second\n"
    assert list(output.parent.iterdir()) == [output]

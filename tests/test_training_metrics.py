import csv
from argparse import ArgumentTypeError

import pytest

from train import (
    _parse_positive_float_or_all,
    _parse_positive_int_or_all,
    build_parser,
)
from taiko_ai.training.metrics import (
    append_episode_metrics,
    make_episode_metrics,
    render_reward_graph,
)


def test_training_cli_accepts_all_tracks_and_full_track_duration() -> None:
    arguments = build_parser().parse_args(
        ["--max-tracks", "all", "--max-seconds", "all"]
    )

    assert arguments.max_tracks is None
    assert arguments.max_seconds is None


def test_training_cli_preserves_bounded_defaults() -> None:
    arguments = build_parser().parse_args([])

    assert arguments.max_tracks == 8
    assert arguments.max_seconds == 30.0
    assert arguments.level is None


def test_training_cli_can_fix_a_single_difficulty() -> None:
    arguments = build_parser().parse_args(["--level", "7"])

    assert arguments.level == 7


@pytest.mark.parametrize("parser,value", [
    (_parse_positive_int_or_all, "0"),
    (_parse_positive_int_or_all, "-1"),
    (_parse_positive_float_or_all, "0"),
    (_parse_positive_float_or_all, "-1"),
    (_parse_positive_float_or_all, "nan"),
    (_parse_positive_float_or_all, "inf"),
])
def test_training_limits_reject_non_positive_values(parser, value) -> None:
    with pytest.raises(ArgumentTypeError):
        parser(value)


def make_metrics(*, episode: int = 1, reward: float = 2.0):
    return make_episode_metrics(
        run_id="test-run",
        episode=episode,
        epoch=1,
        difficulty=5,
        track="song.ogg",
        actions=[0, 1, 2],
        note_onsets=[0.8, 0.1],
        duration=60.0,
        episode_reward=reward,
        average_reward=reward,
        loss=0.5,
    )


def test_episode_metrics_calculate_density_and_onset_accuracy() -> None:
    metrics = make_metrics()

    assert metrics.note_count == 2
    assert metrics.density == 2.0
    assert metrics.onset_accuracy == 0.5


def test_episode_metrics_reject_mismatched_note_onsets() -> None:
    with pytest.raises(ValueError, match="one value for every generated note"):
        make_episode_metrics(
            run_id="test-run",
            episode=1,
            epoch=1,
            difficulty=5,
            track="song.ogg",
            actions=[1, 2],
            note_onsets=[0.8],
            duration=1.0,
            episode_reward=0.0,
            average_reward=0.0,
            loss=0.0,
        )


def test_metrics_append_to_csv_with_a_single_header(tmp_path) -> None:
    path = tmp_path / "logs" / "training.csv"

    append_episode_metrics(path, make_metrics())
    append_episode_metrics(path, make_metrics(episode=2, reward=3.0))

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 2
    assert rows[1]["episode"] == "2"
    assert rows[1]["episode_reward"] == "3.0"


def test_reward_graph_contains_episode_and_moving_average_series() -> None:
    graph = render_reward_graph(
        [make_metrics(episode=1, reward=-1.0), make_metrics(episode=2, reward=2.0)]
    )

    assert "<svg" in graph
    assert "Episode reward" in graph
    assert "Moving average" in graph
    assert "Episode 2" in graph

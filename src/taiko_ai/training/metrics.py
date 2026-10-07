from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from taiko_ai.environment.actions import Action


@dataclass(frozen=True)
class EpisodeMetrics:
    run_id: str
    episode: int
    epoch: int
    difficulty: int
    track: str
    episode_reward: float
    average_reward: float
    note_count: int
    density: float
    onset_accuracy: float
    loss: float


CSV_FIELDS = tuple(EpisodeMetrics.__dataclass_fields__)


def make_episode_metrics(
    *,
    run_id: str,
    episode: int,
    epoch: int,
    difficulty: int,
    track: str,
    actions: list[int],
    note_onsets: list[float],
    duration: float,
    episode_reward: float,
    average_reward: float,
    loss: float,
) -> EpisodeMetrics:
    if duration <= 0:
        raise ValueError("episode duration must be positive")
    note_count = sum(action != Action.REST for action in actions)
    if len(note_onsets) != note_count:
        raise ValueError("note_onsets must contain one value for every generated note")
    onset_accuracy = (
        sum(onset >= 0.35 for onset in note_onsets) / note_count
        if note_count
        else 0.0
    )
    return EpisodeMetrics(
        run_id=run_id,
        episode=episode,
        epoch=epoch,
        difficulty=difficulty,
        track=track,
        episode_reward=episode_reward,
        average_reward=average_reward,
        note_count=note_count,
        density=note_count / duration * 60.0,
        onset_accuracy=onset_accuracy,
        loss=loss,
    )


def append_episode_metrics(path: str | Path, metrics: EpisodeMetrics) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics.__dict__)


def render_reward_graph(metrics: list[EpisodeMetrics]) -> str:
    if not metrics:
        raise ValueError("cannot render a reward graph without episode metrics")

    width, height = 960, 480
    left, right, top, bottom = 72, 24, 40, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    reward_values = [
        value
        for row in metrics
        for value in (row.episode_reward, row.average_reward)
    ]
    low = min(reward_values)
    high = max(reward_values)
    margin = max((high - low) * 0.1, 1.0)
    low -= margin
    high += margin

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * index / max(1, len(metrics) - 1))
        y = top + plot_height * (high - value) / (high - low)
        return x, y

    raw_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for index, row in enumerate(metrics)
        for x, y in [point(index, row.episode_reward)]
    )
    average_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for index, row in enumerate(metrics)
        for x, y in [point(index, row.average_reward)]
    )
    grid = []
    for index in range(5):
        fraction = index / 4
        y = top + plot_height * fraction
        label_value = high - (high - low) * fraction
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
            f'y2="{y:.1f}" stroke="#d1d5db"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12">{label_value:.1f}</text>'
        )
    first_episode = metrics[0].episode
    last_episode = metrics[-1].episode
    run_id = escape(metrics[-1].run_id)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="24" font-size="16" font-weight="bold">'
            f'Training reward (run {run_id})</text>',
            *grid,
            f'<polyline points="{raw_points}" fill="none" stroke="#9ca3af" '
            'stroke-width="1.5"/>',
            f'<polyline points="{average_points}" fill="none" '
            'stroke="#2563eb" stroke-width="2.5"/>',
            f'<text x="{left}" y="{height - 20}" font-size="12">'
            f'Episode {first_episode}</text>',
            f'<text x="{width - right}" y="{height - 20}" text-anchor="end" '
            f'font-size="12">Episode {last_episode}</text>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 3}" '
            'text-anchor="middle" font-size="13">Episode</text>',
            f'<text x="18" y="{top + plot_height / 2:.1f}" '
            'text-anchor="middle" font-size="13" '
            'transform="rotate(-90 18 '
            f'{top + plot_height / 2:.1f})">Reward</text>',
            f'<line x1="{width - 245}" y1="22" x2="{width - 220}" y2="22" '
            'stroke="#9ca3af" stroke-width="2"/>'
            f'<text x="{width - 214}" y="26" font-size="12">Episode reward</text>',
            f'<line x1="{width - 125}" y1="22" x2="{width - 100}" y2="22" '
            'stroke="#2563eb" stroke-width="2.5"/>'
            f'<text x="{width - 94}" y="26" font-size="12">Moving average</text>',
            "</svg>",
            "",
        ]
    )


def save_reward_graph(path: str | Path, metrics: list[EpisodeMetrics]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_reward_graph(metrics), encoding="utf-8")

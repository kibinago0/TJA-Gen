from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

from taiko_ai.agent.ppo_agent import PPOAgent
from taiko_ai.audio.analyzer import AudioFeatures, analyze_audio
from taiko_ai.chart.patterns import HumanPatternModel
from taiko_ai.chart.models import ChartNote, TjaChart
from taiko_ai.chart.repair import repair_chart
from taiko_ai.chart.validator import validate_tja
from taiko_ai.environment.actions import Action
from taiko_ai.environment.difficulty import limit_note_density
from taiko_ai.environment.taiko_env import TaikoEnv
from taiko_ai.export.tja_writer import render_tja


def _wave_reference(audio: Path) -> str:
    return audio.name


def _resolve_timing(
    features: AudioFeatures,
    bpm: float | None,
    offset: float | None,
) -> tuple[AudioFeatures, float]:
    resolved_features = features.with_bpm(bpm) if bpm is not None else features
    return resolved_features, offset if offset is not None else 0.0


def _tja_note_value(action: int) -> int | None:
    return {
        Action.DON: 1,
        Action.KA: 2,
        Action.BIG_DON: 3,
        Action.BIG_KA: 4,
        Action.ROLL_START: 5,
        Action.BALLOON_START: 7,
    }.get(action)


def _parse_difficulty(value: str) -> str:
    difficulties = {
        difficulty.casefold(): difficulty
        for difficulty in ("Easy", "Normal", "Hard", "Oni", "Edit")
    }
    try:
        return difficulties[value.casefold()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "difficulty must be Easy, Normal, Hard, Oni, or Edit"
        ) from error


def _parse_positive_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive finite number"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "value must be a positive finite number"
        )
    return parsed


def _parse_finite(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be finite") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _validate_args(args: argparse.Namespace) -> None:
    if args.audio.suffix.casefold() not in {".wav", ".ogg", ".flac"}:
        raise ValueError("audio file must be WAV, OGG, or FLAC")
    if not 1 <= args.level <= 10:
        raise ValueError("level must be between 1 and 10")
    if args.bpm is not None and (
        not math.isfinite(args.bpm) or args.bpm <= 0
    ):
        raise ValueError("BPM must be a positive finite number")
    if args.offset is not None and not math.isfinite(args.offset):
        raise ValueError("OFFSET must be finite")
    if args.max_seconds is not None and (
        not math.isfinite(args.max_seconds) or args.max_seconds <= 0
    ):
        raise ValueError("max-seconds must be a positive finite number")
    if args.maximum_density is not None and (
        not math.isfinite(args.maximum_density) or args.maximum_density <= 0
    ):
        raise ValueError("maximum-density must be a positive finite number")
    if args.output.suffix.casefold() != ".tja":
        raise ValueError("output path must use the .tja extension")
    output_path = args.output.resolve()
    if output_path in (args.audio.resolve(), args.model.resolve()):
        raise ValueError("output path must not overwrite the input audio or model")


def _write_tja_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(contents)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def generate(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"PPO model not found: {args.model}. Train one first with train.py."
        )

    features = analyze_audio(args.audio, max_duration=args.max_seconds)
    features, offset = _resolve_timing(features, args.bpm, args.offset)
    agent = PPOAgent(device=args.device)
    metadata = agent.load(args.model)
    raw_pattern_model = metadata.get("human_pattern_model")
    if raw_pattern_model is not None and not isinstance(raw_pattern_model, dict):
        raise ValueError("checkpoint human pattern model metadata is invalid")
    pattern_model = (
        HumanPatternModel.from_dict(raw_pattern_model)
        if raw_pattern_model is not None
        else None
    )
    env = TaikoEnv(
        features,
        difficulty=args.level,
        pattern_model=pattern_model,
    )
    observation = env.reset()
    notes: list[ChartNote] = []
    last_note_time: float | None = None
    while True:
        action = agent.select_action(observation, deterministic=True)
        time = env.index * env.step_duration
        if time <= env.special_until + env.step_duration * 0.5:
            action = int(Action.REST)
        else:
            action = limit_note_density(action, time, last_note_time, args.level)
        observation, _, done, info = env.step(action)
        note_value = _tja_note_value(action)
        if note_value is not None:
            duration = (
                float(info["special_duration"])
                if action in (Action.ROLL_START, Action.BALLOON_START)
                else 0.0
            )
            if float(info["time"]) + duration > features.duration:
                note_value = (
                    1 if action == Action.ROLL_START else 2
                )
                duration = 0.0
                env.special_until = float(info["time"])
            balloon_hits = (
                max(5, round(duration * (12 + args.level * 2)))
                if action == Action.BALLOON_START and duration > 0.0
                else None
            )
            notes.append(
                ChartNote(
                    time=float(info["time"]),
                    value=note_value,
                    duration=duration,
                    balloon_hits=balloon_hits,
                )
            )
            last_note_time = float(info["time"])
        if done:
            break

    chart = TjaChart(
        title=args.audio.stem.replace("\r", " ").replace("\n", " "),
        bpm=features.bpm,
        course=args.difficulty,
        level=args.level,
        notes=notes,
        offset=offset,
        wave=_wave_reference(args.audio),
    )
    repair = repair_chart(
        chart,
        duration=features.duration,
        minimum_interval=env.step_duration - 1e-6,
        maximum_density=args.maximum_density,
    )
    chart = repair.chart
    for change in repair.changes:
        print(f"Chart repair: {change}")
    rendered = render_tja(chart)
    errors = validate_tja(
        rendered,
        duration=features.duration,
        minimum_interval=env.step_duration - 1e-6,
        maximum_density=args.maximum_density,
    )
    if errors:
        raise ValueError("Generated chart is invalid: " + "; ".join(errors))

    _write_tja_atomically(args.output, rendered)
    print(
        f"Wrote {args.output} ({len(chart.notes)} notes, "
        f"BPM {features.bpm:.1f}, OFFSET {offset:g}, {features.duration:.1f}s)"
    )
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Taiko TJA chart from a WAV, OGG, or FLAC file."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("models/ppo_taiko.pt")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--difficulty", type=_parse_difficulty, default="Oni")
    parser.add_argument("--level", type=int, choices=range(1, 11), default=5)
    parser.add_argument(
        "--bpm",
        type=_parse_positive_finite,
        help="override the automatically analyzed BPM",
    )
    parser.add_argument(
        "--offset",
        type=_parse_finite,
        help="override OFFSET (defaults to 0.0; audio starts at time zero)",
    )
    parser.add_argument("--max-seconds", type=_parse_positive_finite)
    parser.add_argument(
        "--maximum-density",
        type=_parse_positive_finite,
        help="optional maximum count of playable notes per second",
    )
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.output is None:
        arguments.output = Path("output") / f"{arguments.audio.stem}.tja"
    generate(arguments)

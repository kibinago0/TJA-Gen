from __future__ import annotations

import argparse
import math
import random
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from taiko_ai.agent.ppo_agent import PPOAgent, Transition
from taiko_ai.audio.analyzer import analyze_audio
from taiko_ai.chart.patterns import HumanPatternModel
from taiko_ai.chart.tja_parser import load_tja
from taiko_ai.environment.taiko_env import TaikoEnv
from taiko_ai.environment.actions import Action
from taiko_ai.training.metrics import (
    EpisodeMetrics,
    append_episode_metrics,
    make_episode_metrics,
    save_reward_graph,
)


AUDIO_EXTENSIONS = {".wav", ".ogg", ".flac"}


def _parse_positive_int_or_all(value: str) -> int | None:
    if value.casefold() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer or 'all'")
    return parsed


def _parse_positive_float_or_all(value: str) -> float | None:
    if value.casefold() == "all":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be positive or 'all'") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive or 'all'")
    return parsed


def find_audio_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Audio dataset directory does not exist: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in AUDIO_EXTENSIONS
        and path.with_suffix(".tja").is_file()
    )


def build_human_pattern_model(directory: Path) -> HumanPatternModel:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Human pattern chart directory does not exist: {directory}"
        )
    model = HumanPatternModel()
    chart_paths = sorted(directory.rglob("*.tja"))
    for chart_path in chart_paths:
        model.add_chart(load_tja(chart_path))
    if model.chart_count == 0 or model.note_count == 0:
        raise ValueError("No chart notes available to build the human pattern model")
    return model


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tracks = find_audio_files(args.data_dir)
    if not tracks:
        raise ValueError(f"No supported audio files found under {args.data_dir}")
    pattern_data_dir = args.pattern_data_dir or args.data_dir
    pattern_model = build_human_pattern_model(pattern_data_dir)
    if args.max_tracks is not None:
        tracks = tracks[: args.max_tracks]
    agent = PPOAgent(device=args.device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics_history: list[EpisodeMetrics] = []
    recent_rewards: deque[float] = deque(maxlen=20)
    episode_number = 0
    print(f"Training on {len(tracks)} audio files from {args.data_dir}")
    for epoch in range(args.epochs):
        random.shuffle(tracks)
        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        for track in tracks:
            chart = load_tja(track.with_suffix(".tja"))
            features = analyze_audio(track, max_duration=args.max_seconds)
            features = features.with_bpm(chart.bpm)
            difficulty = args.level if args.level is not None else random.randint(1, 10)
            env = TaikoEnv(
                features,
                difficulty=difficulty,
                pattern_model=pattern_model,
            )
            observation = env.reset()
            transitions: list[Transition] = []
            total_reward = 0.0
            actions: list[int] = []
            note_onsets: list[float] = []
            while True:
                action, log_probability, value = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                transitions.append(
                    Transition(
                        observation=observation,
                        action=action,
                        log_probability=log_probability,
                        reward=reward,
                        value=value,
                        done=done,
                    )
                )
                total_reward += reward
                actions.append(action)
                if action != Action.REST:
                    note_onsets.append(float(info["onset"]))
                observation = next_observation
                if done:
                    break
            loss = agent.update(transitions)
            epoch_rewards.append(total_reward)
            epoch_losses.append(loss)
            episode_number += 1
            recent_rewards.append(total_reward)
            metrics = make_episode_metrics(
                run_id=run_id,
                episode=episode_number,
                epoch=epoch + 1,
                difficulty=difficulty,
                track=track.name,
                actions=actions,
                note_onsets=note_onsets,
                duration=features.duration,
                episode_reward=total_reward,
                average_reward=float(np.mean(recent_rewards)),
                loss=loss,
            )
            metrics_history.append(metrics)
            append_episode_metrics(args.log_file, metrics)
            print(
                f"Episode {episode_number}: reward={metrics.episode_reward:.2f}, "
                f"average_reward={metrics.average_reward:.2f}, "
                f"level={metrics.difficulty}, "
                f"notes={metrics.note_count}, density={metrics.density:.2f}/min, "
                f"onset_accuracy={metrics.onset_accuracy:.1%}"
            )
        mean_reward = float(np.mean(epoch_rewards))
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"mean_reward={mean_reward:.2f}, "
            f"loss={float(np.mean(epoch_losses)):.4f}"
        )
        save_reward_graph(args.graph_file, metrics_history)

    agent.save(
        args.output,
        metadata={"human_pattern_model": pattern_model.to_dict()},
    )
    print(f"Saved PPO model to {args.output}")
    print(
        f"Human pattern model: {pattern_model.chart_count} charts, "
        f"{pattern_model.note_count} playable notes from {pattern_data_dir}"
    )
    print(f"Saved training metrics to {args.log_file}")
    print(f"Saved reward graph to {args.graph_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PPO chart generator using audio-derived rewards."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "TJA-Gen-data",
        help="directory containing WAV, OGG, or FLAC audio files",
    )
    parser.add_argument(
        "--pattern-data-dir",
        type=Path,
        default=None,
        help="directory containing human TJA charts (defaults to --data-dir)",
    )
    parser.add_argument("--output", type=Path, default=Path("models/ppo_taiko.pt"))
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/training.csv"),
        help="append per-episode metrics to this CSV file",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=Path("graphs/training.svg"),
        help="write the current run's reward graph to this SVG file",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-tracks",
        type=_parse_positive_int_or_all,
        default=8,
        help="maximum number of paired tracks to train on, or 'all'",
    )
    parser.add_argument(
        "--max-seconds",
        type=_parse_positive_float_or_all,
        default=30.0,
        help="maximum audio duration per track in seconds, or 'all'",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=range(1, 11),
        default=None,
        help=(
            "fixed target difficulty; by default, sample a level from 1 to 10 "
            "for every episode"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    train(arguments)

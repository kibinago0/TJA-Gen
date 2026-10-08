import random
from collections import Counter

import pytest

from taiko_ai.agent.ppo_agent import PPOAgent
from taiko_ai.chart.models import ChartNote, TjaChart
from taiko_ai.chart.patterns import HumanPatternModel
from taiko_ai.environment.actions import Action
from taiko_ai.environment.state import OBSERVATION_SIZE


def make_chart(values: list[int]) -> TjaChart:
    return TjaChart(
        title="Patterns",
        bpm=120.0,
        course="Oni",
        level=5,
        notes=[
            ChartNote(time=index * 0.25, value=value)
            for index, value in enumerate(values)
        ],
    )


def test_human_pattern_model_rewards_common_and_penalizes_unseen_patterns() -> None:
    model = HumanPatternModel()
    for _ in range(12):
        model.add_chart(make_chart([1, 1, 1, 1]))
        model.add_chart(make_chart([2, 2, 2, 2]))

    assert model.score([Action.DON], Action.DON) == 5.0
    assert model.score([Action.DON], Action.KA) == -5.0
    assert model.score([Action.REST], Action.DON) == 0.0


def test_pattern_frequency_guides_reward_and_note_sampling() -> None:
    model = HumanPatternModel(counts=Counter({"DD": 90, "DK": 10}))

    probabilities = model.next_symbol_probabilities([Action.DON])
    sampled = [
        model.sample_action(Action.DON, [Action.DON], random.Random(seed))
        for seed in range(1000)
    ]

    assert probabilities[0] > probabilities[1]
    assert model.score([Action.DON], Action.DON) == 5.0
    assert model.score([Action.DON], Action.KA) == -5.0
    assert 850 < sampled.count(Action.DON) < 950
    assert 50 < sampled.count(Action.KA) < 150


def test_pattern_sampling_preserves_big_notes_and_other_actions() -> None:
    model = HumanPatternModel(counts=Counter({"DD": 90, "DK": 10}))

    assert (
        model.sample_action(
            Action.BIG_DON,
            [Action.DON],
            random.Random(7),
        )
        == Action.BIG_DON
    )
    assert (
        model.sample_action(Action.ROLL_START, [], random.Random(7))
        == Action.ROLL_START
    )


def test_pattern_sampling_without_chart_data_keeps_the_policy_action() -> None:
    action = HumanPatternModel().sample_action(
        Action.DON,
        [Action.DON],
        random.Random(7),
    )

    assert action == Action.DON


def test_pattern_model_round_trips_and_rejects_invalid_counts() -> None:
    model = HumanPatternModel()
    model.add_chart(make_chart([1, 2, 1, 2, 1]))

    restored = HumanPatternModel.from_dict(model.to_dict())

    assert restored.counts == model.counts
    assert restored.chart_count == 1
    assert restored.note_count == 5
    with pytest.raises(ValueError, match="invalid pattern counts"):
        HumanPatternModel.from_dict(
            {"counts": {"DX": 4}, "chart_count": 1, "note_count": 4}
        )


def test_ppo_checkpoint_round_trips_human_pattern_metadata(tmp_path) -> None:
    model = HumanPatternModel()
    model.add_chart(make_chart([1, 1, 2, 2]))
    agent = PPOAgent()
    checkpoint = tmp_path / "model.pt"

    agent.save(checkpoint, metadata={"human_pattern_model": model.to_dict()})
    metadata = PPOAgent().load(checkpoint)
    restored = HumanPatternModel.from_dict(metadata["human_pattern_model"])

    assert restored.counts == model.counts
    assert restored.chart_count == 1
    assert OBSERVATION_SIZE == 36
    assert metadata_action_count(checkpoint) == 7


def test_ppo_rejects_checkpoint_with_old_observation_shape(tmp_path) -> None:
    agent = PPOAgent()
    checkpoint = tmp_path / "old-model.pt"
    agent.save(checkpoint)

    import torch

    data = torch.load(checkpoint, weights_only=True)
    data["observation_size"] = 9
    torch.save(data, checkpoint)

    with pytest.raises(ValueError, match="retrain the model"):
        PPOAgent().load(checkpoint)


def metadata_action_count(checkpoint):
    import torch

    return torch.load(checkpoint, weights_only=True)["action_count"]


def test_ppo_rejects_checkpoint_with_old_action_count(tmp_path) -> None:
    agent = PPOAgent()
    checkpoint = tmp_path / "old-actions.pt"
    agent.save(checkpoint)

    import torch

    data = torch.load(checkpoint, weights_only=True)
    data["action_count"] = 3
    torch.save(data, checkpoint)

    with pytest.raises(ValueError, match="action count.*retrain the model"):
        PPOAgent().load(checkpoint)


def test_chart_pattern_count_maps_big_notes_to_hand_type_and_skips_rolls() -> None:
    model = HumanPatternModel()
    model.add_chart(make_chart([3, 4, 5, 1]))

    assert model.counts["DK"] == 1
    assert model.counts["KD"] == 1
    assert model.note_count == 3


def test_pattern_model_maps_big_actions_to_their_hand_types() -> None:
    model = HumanPatternModel()
    for _ in range(12):
        model.add_chart(make_chart([1, 1, 1, 1]))

    assert model.score([Action.BIG_DON], Action.BIG_DON) == 5.0

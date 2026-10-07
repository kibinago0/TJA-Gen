from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from taiko_ai.chart.models import TjaChart
from taiko_ai.environment.actions import Action


@dataclass
class HumanPatternModel:
    counts: Counter[str] = field(default_factory=Counter)
    chart_count: int = 0
    note_count: int = 0
    maximum_length: int = 4

    def add_chart(self, chart: TjaChart) -> None:
        hits = [
            "D" if note.value in (1, 3) else "K"
            for note in chart.notes
            if note.value in (1, 2, 3, 4)
        ]
        self.chart_count += 1
        self.note_count += len(hits)
        for length in range(2, self.maximum_length + 1):
            self.counts.update(
                "".join(hits[start : start + length])
                for start in range(len(hits) - length + 1)
            )

    def score(self, history: Iterable[int], action: int) -> float:
        symbol = _action_symbol(action)
        if symbol is None or not self.counts:
            return 0.0

        previous_symbols = [
            history_symbol
            for item in history
            if (history_symbol := _action_symbol(item)) is not None
        ]
        previous = "".join(previous_symbols)
        for context_length in range(
            min(self.maximum_length - 1, len(previous)), 0, -1
        ):
            context = previous[-context_length:]
            drum_count = self.counts[context + "D"]
            ka_count = self.counts[context + "K"]
            context_count = drum_count + ka_count
            if context_count < 10:
                continue
            count = drum_count if symbol == "D" else ka_count
            probability = (count + 1.0) / (context_count + 2.0)
            if count >= 10 and probability >= 0.2:
                return 5.0
            if count >= 3 and probability >= 0.05:
                return 2.0
            return -5.0
        return 0.0

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "counts": dict(self.counts),
            "chart_count": self.chart_count,
            "note_count": self.note_count,
            "maximum_length": self.maximum_length,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HumanPatternModel:
        raw_counts = data.get("counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("pattern model is missing its counts")
        counts: Counter[str] = Counter()
        for pattern, count in raw_counts.items():
            if (
                not isinstance(pattern, str)
                or not pattern
                or any(symbol not in "DK" for symbol in pattern)
                or len(pattern) > 4
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("pattern model contains invalid pattern counts")
            counts[pattern] = count
        chart_count = data.get("chart_count", 0)
        note_count = data.get("note_count", 0)
        maximum_length = data.get("maximum_length", 4)
        if (
            not isinstance(chart_count, int)
            or chart_count < 0
            or not isinstance(note_count, int)
            or note_count < 0
            or maximum_length != 4
        ):
            raise ValueError("pattern model metadata is invalid")
        return cls(
            counts=counts,
            chart_count=chart_count,
            note_count=note_count,
            maximum_length=maximum_length,
        )


def _action_symbol(action: int) -> str | None:
    if action in (Action.DON, Action.BIG_DON):
        return "D"
    if action in (Action.KA, Action.BIG_KA):
        return "K"
    return None

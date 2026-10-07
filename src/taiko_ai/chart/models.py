from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChartNote:
    time: float
    value: int
    duration: float = 0.0
    balloon_hits: int | None = None
    beat_position: float | None = field(default=None, compare=False)
    duration_beats: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class ChartEvent:
    time: float
    command: str
    value: float
    beat_position: float | None = field(default=None, compare=False)


@dataclass
class TjaChart:
    title: str
    bpm: float
    course: str
    level: int
    notes: list[ChartNote] = field(default_factory=list)
    offset: float = 0.0
    wave: str = ""
    events: list[ChartEvent] = field(default_factory=list)

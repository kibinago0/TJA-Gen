from __future__ import annotations

from math import isfinite

from taiko_ai.chart.tja_parser import parse_tja
from taiko_ai.chart.models import TjaChart


def validate_chart(
    chart: TjaChart,
    duration: float | None = None,
    minimum_interval: float = 0.0,
    maximum_density: float | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isfinite(chart.bpm) or chart.bpm <= 0:
        errors.append("BPM must be positive")
        if not isfinite(chart.bpm):
            errors.append("BPM must be finite")
    if not 1 <= chart.level <= 10:
        errors.append("LEVEL must be between 1 and 10")
    if not chart.notes:
        errors.append("chart contains no notes")
    if not isfinite(chart.offset):
        errors.append("OFFSET must be finite")
    if not isfinite(minimum_interval) or minimum_interval < 0:
        errors.append("minimum_interval must be finite and non-negative")
    if maximum_density is not None and (
        not isfinite(maximum_density) or maximum_density <= 0
    ):
        errors.append("maximum_density must be finite and positive")
    if duration is not None and (not isfinite(duration) or duration <= 0):
        errors.append("audio duration must be finite and positive")

    previous_event_time = -1.0
    for event in chart.events:
        command = event.command.upper()
        if command not in {"BPMCHANGE", "SCROLL", "DELAY"}:
            errors.append(f"unsupported chart event: {event.command}")
        if not isfinite(event.time) or event.time < 0:
            errors.append(f"#{command} time must be finite and non-negative")
        if not isfinite(event.value):
            errors.append(f"#{command} value must be finite")
        elif command == "BPMCHANGE" and event.value <= 0:
            errors.append("#BPMCHANGE must be positive")
        if isfinite(event.time):
            if event.time < previous_event_time:
                errors.append("event times must be in ascending order")
            previous_event_time = event.time
        if command == "DELAY" and event.value < 0:
            errors.append("negative #DELAY is not supported")
        if event.beat_position is not None and (
            not isfinite(event.beat_position) or event.beat_position < 0
        ):
            errors.append(f"#{command} beat position must be finite and non-negative")

    previous_time = -1.0
    playable_note_count = 0
    for note in chart.notes:
        if note.value not in range(1, 10):
            errors.append(f"invalid note value: {note.value}")
        if not isfinite(note.time):
            errors.append("note time must be finite")
        if not isfinite(note.duration) or note.duration < 0:
            errors.append("note duration must be finite and non-negative")
        if note.beat_position is not None and (
            not isfinite(note.beat_position) or note.beat_position < 0
        ):
            errors.append("note beat position must be finite and non-negative")
        if note.duration_beats is not None and (
            not isfinite(note.duration_beats) or note.duration_beats <= 0
        ):
            errors.append("special-note beat duration must be finite and positive")
        if note.value in (5, 7):
            playable_note_count += 1
            if note.duration <= 0:
                errors.append("roll and balloon notes must have positive duration")
            if duration is not None and note.time + note.duration > duration:
                errors.append("roll or balloon extends beyond audio duration")
            if note.value == 7:
                if note.balloon_hits is None or note.balloon_hits <= 0:
                    errors.append("balloon must have a positive hit count")
        elif note.value == 8:
            errors.append("unmatched roll/balloon end marker")
        else:
            playable_note_count += 1
        if note.value not in (5, 7) and (
            note.duration != 0
            or note.balloon_hits is not None
            or note.duration_beats is not None
        ):
            errors.append("only roll and balloon notes can have duration metadata")
        if isfinite(note.time) and note.time < 0:
            errors.append("note time must not be negative")
        if isfinite(note.time) and note.time < previous_time:
            errors.append("note times must be in ascending order")
        if (
            isfinite(note.time)
            and previous_time >= 0
            and note.time - previous_time < minimum_interval
        ):
            errors.append("note interval too short")
        if duration is not None and isfinite(note.time) and note.time >= duration:
            errors.append("note time exceeds audio duration")
        if isfinite(note.time):
            previous_time = note.time

    if (
        maximum_density is not None
        and duration is not None
        and duration > 0
        and playable_note_count / duration > maximum_density
    ):
        errors.append("note density exceeds maximum_density")
    return errors


def validate_tja(
    source: str,
    course: str | None = None,
    *,
    duration: float | None = None,
    minimum_interval: float = 0.0,
    maximum_density: float | None = None,
) -> list[str]:
    try:
        chart = parse_tja(source, course=course)
    except (ValueError, OverflowError) as error:
        return [f"invalid TJA syntax: {error}"]
    return validate_chart(
        chart,
        duration=duration,
        minimum_interval=minimum_interval,
        maximum_density=maximum_density,
    )

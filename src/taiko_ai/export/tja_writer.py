from __future__ import annotations

from math import isfinite

from taiko_ai.chart.models import ChartEvent, ChartNote, TjaChart


def _slot_at_time(
    time: float,
    events: list[ChartEvent],
    initial_bpm: float,
    subdivisions: int,
) -> int:
    cursor_time = 0.0
    slot = 0.0
    bpm = initial_bpm
    for event in events:
        if event.time > time:
            break
        slot += max(0.0, event.time - cursor_time) * bpm * subdivisions / 240.0
        cursor_time = event.time
        if event.command == "BPMCHANGE":
            bpm = event.value
        elif event.command == "DELAY":
            if event.value < 0:
                raise ValueError("negative #DELAY cannot be exported reliably")
            cursor_time += event.value
    slot += max(0.0, time - cursor_time) * bpm * subdivisions / 240.0
    return int(round(slot))


def _validate_event(event: ChartEvent) -> None:
    command = event.command.upper()
    if command not in {"BPMCHANGE", "SCROLL", "DELAY"}:
        raise ValueError(f"unsupported chart event: {event.command}")
    if not isfinite(event.time) or event.time < 0:
        raise ValueError(f"#{command} time must be finite and non-negative")
    if not isfinite(event.value):
        raise ValueError(f"#{command} value must be finite")
    if command == "BPMCHANGE" and event.value <= 0:
        raise ValueError("#BPMCHANGE must be positive")


def render_tja(chart: TjaChart, subdivisions: int = 16) -> str:
    if chart.bpm <= 0:
        raise ValueError("BPM must be positive")
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")

    slots: dict[int, int] = {}
    balloon_hits: list[int] = []
    events = sorted(chart.events, key=lambda event: event.time)
    for event in events:
        _validate_event(event)
    event_slots: dict[int, list[ChartEvent]] = {}
    for event in events:
        slot = (
            int(round(event.beat_position * subdivisions / 4.0))
            if event.beat_position is not None
            else _slot_at_time(event.time, events, chart.bpm, subdivisions)
        )
        event_slots.setdefault(slot, []).append(event)

    for note in chart.notes:
        index = (
            int(round(note.beat_position * subdivisions / 4.0))
            if note.beat_position is not None
            else _slot_at_time(note.time, events, chart.bpm, subdivisions)
        )
        if index in slots:
            raise ValueError(f"multiple notes map to TJA slot {index}")
        if note.value not in range(1, 10):
            raise ValueError(f"invalid TJA note value: {note.value}")
        slots[index] = note.value
        if note.value in (5, 7):
            if note.duration <= 0:
                raise ValueError("roll and balloon notes require a positive duration")
            end_index = (
                int(
                    round(
                        (note.beat_position + note.duration_beats)
                        * subdivisions
                        / 4.0
                    )
                )
                if note.beat_position is not None and note.duration_beats is not None
                else _slot_at_time(
                    note.time + note.duration,
                    events,
                    chart.bpm,
                    subdivisions,
                )
            )
            if end_index <= index:
                raise ValueError("roll and balloon duration is shorter than one TJA slot")
            if end_index in slots:
                raise ValueError("roll/balloon end marker overlaps another note")
            slots[end_index] = 8
            if note.value == 7:
                if note.balloon_hits is None or note.balloon_hits <= 0:
                    raise ValueError("balloon notes require a positive hit count")
                balloon_hits.append(note.balloon_hits)
        elif note.duration != 0 or note.balloon_hits is not None:
            raise ValueError("only roll and balloon notes can have duration metadata")

    measure_count = max(
        1,
        max([*slots, *event_slots], default=0) // subdivisions + 1,
    )
    lines = [
        f"TITLE:{chart.title}",
        f"BPM:{chart.bpm:g}",
        f"WAVE:{chart.wave}",
        f"OFFSET:{chart.offset:g}",
        "",
        f"COURSE:{chart.course}",
        f"LEVEL:{chart.level}",
    ]
    if balloon_hits:
        lines.append("#BALLOON:" + ",".join(map(str, balloon_hits)))
    lines.extend(
        [
            "",
            "#START",
        ]
    )
    for measure_index in range(measure_count):
        start = measure_index * subdivisions
        position = 0
        for relative_slot in range(subdivisions):
            absolute_slot = start + relative_slot
            for event in event_slots.get(absolute_slot, []):
                if relative_slot > position:
                    lines.append(
                        "".join(
                            str(slots.get(start + slot, 0))
                            for slot in range(position, relative_slot)
                        )
                    )
                    position = relative_slot
                lines.append(f"#{event.command.upper()} {event.value:g}")
        lines.append(
            "".join(
                str(slots.get(start + slot, 0))
                for slot in range(position, subdivisions)
            )
            + ","
        )
    lines.extend(("#END", ""))
    return "\n".join(lines)

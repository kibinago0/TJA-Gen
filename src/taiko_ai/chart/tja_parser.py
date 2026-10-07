from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from taiko_ai.chart.models import ChartEvent, ChartNote, TjaChart


_HEADER = re.compile(r"^\s*([^:#]+)\s*:\s*(.*?)\s*$")
_COMMAND = re.compile(r"^\s*#([A-Z]+)(?:\s+(.+?))?\s*$", re.IGNORECASE)
_BALLOON = re.compile(r"^\s*#?BALLOON\s*:\s*(.*?)\s*$", re.IGNORECASE)


def _parse_chart_block(
    lines: list[str],
    headers: dict[str, str],
    default_bpm: float,
) -> TjaChart:
    bpm = float(headers.get("BPM", default_bpm))
    chart_bpm = bpm
    notes: list[ChartNote] = []
    events: list[ChartEvent] = []
    elapsed = 0.0
    beat_position = 0.0
    measure_digits = ""
    measure_commands: list[tuple[int, str, str]] = []

    def add_measure() -> None:
        nonlocal elapsed, beat_position, bpm, measure_digits, measure_commands
        digits = measure_digits or "0"

        commands_by_position: dict[int, list[tuple[str, str]]] = {}
        for position, name, argument in measure_commands:
            commands_by_position.setdefault(position, []).append((name, argument))

        measure_start_beats = beat_position
        last_position = 0
        for position in range(len(digits) + 1):
            elapsed += 4.0 * (position - last_position) / len(digits) * 60.0 / bpm
            last_position = position
            note_beat_position = measure_start_beats + 4.0 * position / len(digits)
            for name, argument in commands_by_position.get(position, []):
                try:
                    value = float(argument.split()[0])
                except (ValueError, IndexError) as error:
                    raise ValueError(f"invalid #{name} value: {argument!r}") from error
                if name == "BPMCHANGE":
                    if value <= 0:
                        raise ValueError("#BPMCHANGE must be positive")
                    events.append(
                        ChartEvent(
                            elapsed,
                            name,
                            value,
                            note_beat_position,
                        )
                    )
                    bpm = value
                elif name == "SCROLL":
                    events.append(
                        ChartEvent(
                            elapsed,
                            name,
                            value,
                            note_beat_position,
                        )
                    )
                elif name == "DELAY":
                    events.append(
                        ChartEvent(
                            elapsed,
                            name,
                            value,
                            note_beat_position,
                        )
                    )
                    elapsed += value

            if position < len(digits) and digits[position] != "0":
                notes.append(
                    ChartNote(
                        elapsed,
                        int(digits[position]),
                        beat_position=note_beat_position,
                    )
                )

        beat_position = measure_start_beats + 4.0
        measure_digits = ""
        measure_commands = []

    for raw_line in lines:
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith("//"):
            continue
        command = _COMMAND.match(line)
        if command:
            name = command.group(1).upper()
            argument = command.group(2)
            if name in ("BPMCHANGE", "SCROLL", "DELAY"):
                if not argument:
                    raise ValueError(f"#{name} requires a value")
                measure_commands.append((len(measure_digits), name, argument))
            continue
        if line.startswith("#"):
            continue
        if ":" in line:
            continue
        if "," in line:
            before, _, remainder = line.partition(",")
            measure_digits += _validate_measure_content(before)
            add_measure()
            measure_digits += _validate_measure_content(remainder)
        else:
            measure_digits += _validate_measure_content(line)

    if measure_digits.strip() or measure_commands:
        add_measure()

    balloon_header = headers.get("BALLOON", "").split("//", 1)[0]
    balloon_values = [
        int(value.strip())
        for value in balloon_header.split(",")
        if value.strip()
    ]
    balloon_index = 0
    open_specials: list[int] = []
    remove_markers: set[int] = set()
    for index, note in enumerate(notes):
        if note.value in (5, 7):
            if note.value == 7:
                if balloon_index < len(balloon_values):
                    balloon_hits = balloon_values[balloon_index]
                    balloon_index += 1
                    note = replace(note, balloon_hits=balloon_hits)
                    notes[index] = note
            open_specials.append(index)
        elif note.value == 8 and open_specials:
            start_index = open_specials.pop()
            start_note = notes[start_index]
            duration = note.time - start_note.time
            if duration <= 0:
                if (
                    note.beat_position is None
                    or start_note.beat_position is None
                    or note.beat_position <= start_note.beat_position
                ):
                    raise ValueError("roll/balloon end marker must follow its start")
            duration_beats = (
                note.beat_position - start_note.beat_position
                if note.beat_position is not None
                and start_note.beat_position is not None
                else None
            )
            if duration <= 0 and duration_beats is not None:
                duration = duration_beats * 60.0 / chart_bpm
            notes[start_index] = replace(
                start_note,
                duration=duration,
                duration_beats=duration_beats,
            )
            remove_markers.add(index)
    notes = [note for index, note in enumerate(notes) if index not in remove_markers]

    return TjaChart(
        title=headers.get("TITLE", "Generated"),
        bpm=chart_bpm,
        course=headers.get("COURSE", "Oni"),
        level=int(float(headers.get("LEVEL", "1"))),
        notes=notes,
        offset=float(headers.get("OFFSET", "0")),
        wave=headers.get("WAVE", ""),
        events=events,
    )


def parse_tja(
    text: str,
    course: str | None = None,
) -> TjaChart:
    headers: dict[str, str] = {}
    selected: TjaChart | None = None
    chart_headers: dict[str, str] = {}
    chart_lines: list[str] = []
    in_chart = False

    def finish_chart() -> None:
        nonlocal selected, chart_headers, chart_lines
        chart = _parse_chart_block(
            chart_lines,
            {**headers, **chart_headers},
            float(headers.get("BPM", "120")),
        )
        if selected is None and (course is None or chart.course.casefold() == course.casefold()):
            selected = chart
        chart_headers = {}
        chart_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        command = _COMMAND.match(line)
        if command:
            name = command.group(1).upper()
            if name == "START":
                in_chart = True
            elif name == "END" and in_chart:
                finish_chart()
                in_chart = False
            elif in_chart:
                chart_lines.append(line)
            continue

        if in_chart:
            chart_lines.append(line)
            continue

        balloon = _BALLOON.match(line)
        if balloon:
            headers["BALLOON"] = balloon.group(1)
            chart_headers["BALLOON"] = balloon.group(1)
            continue

        header = _HEADER.match(line)
        if header:
            key, value = header.group(1).strip().upper(), header.group(2).strip()
            headers[key] = value
            chart_headers[key] = value

    if in_chart:
        finish_chart()
    if selected is None:
        requested = f" for course {course!r}" if course else ""
        raise ValueError(f"No matching #START chart found{requested}")
    return selected


def _validate_measure_content(content: str) -> str:
    digits: list[str] = []
    for character in content:
        if character.isspace():
            continue
        if not character.isdigit():
            raise ValueError(f"invalid character in TJA note data: {character!r}")
        digits.append(character)
    return "".join(digits)


def load_tja(path: str | Path, course: str | None = None) -> TjaChart:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    return parse_tja(text, course=course)

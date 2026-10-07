from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

from taiko_ai.chart.models import ChartNote, TjaChart
from taiko_ai.chart.validator import validate_chart


@dataclass(frozen=True)
class RepairReport:
    chart: TjaChart
    changes: tuple[str, ...]


def repair_chart(
    chart: TjaChart,
    *,
    duration: float | None = None,
    minimum_interval: float = 0.0,
    maximum_density: float | None = None,
    subdivisions: int = 16,
) -> RepairReport:
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive")
    if not isfinite(chart.bpm) or chart.bpm <= 0:
        raise ValueError("cannot repair a chart with a non-positive or non-finite BPM")
    if not isfinite(minimum_interval) or minimum_interval < 0:
        raise ValueError("minimum_interval must be finite and non-negative")
    if duration is not None and (not isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    if maximum_density is not None and (
        not isfinite(maximum_density) or maximum_density <= 0
    ):
        raise ValueError("maximum_density must be finite and positive")

    changes: list[str] = []
    ordered_notes = list(enumerate(chart.notes))
    ordered_notes.sort(
        key=lambda indexed: (
            indexed[1].time
            if isfinite(indexed[1].time)
            else float("inf"),
            indexed[0],
        )
    )
    slot_seconds = 240.0 / chart.bpm / subdivisions
    time_slots: dict[int, ChartNote] = {}
    retained: list[ChartNote] = []

    for original_index, note in ordered_notes:
        if not 1 <= note.value <= 7:
            if note.value != 8:
                raise ValueError(f"cannot repair unsupported note value {note.value}")
            changes.append(f"Removed unmatched end marker at note {original_index + 1}")
            continue
        if not isfinite(note.time) or note.time < 0:
            changes.append(f"Removed note with invalid time at note {original_index + 1}")
            continue
        if note.value in (5, 7):
            if (
                not isfinite(note.duration)
                or note.duration <= 0
                or (note.value == 7 and (note.balloon_hits is None or note.balloon_hits <= 0))
            ):
                changes.append(
                    f"Removed invalid special note at {note.time:.3f}s"
                )
                continue

        time = note.time
        if duration is not None and time >= duration:
            changes.append(f"Removed note beyond audio duration at {time:.3f}s")
            continue
        if time != 0 and note.beat_position is None:
            snapped_time = round(time / slot_seconds) * slot_seconds
            if abs(snapped_time - time) > 1e-9:
                changes.append(f"Snapped note from {time:.3f}s to {snapped_time:.3f}s")
            time = snapped_time
        if duration is not None and time >= duration:
            changes.append(f"Removed note beyond audio duration at {time:.3f}s")
            continue

        note_duration = note.duration
        if note.value in (5, 7) and duration is not None:
            available_duration = duration - time
            if note_duration > available_duration:
                if available_duration <= 0:
                    changes.append(f"Removed special note beyond audio duration at {time:.3f}s")
                    continue
                note_duration = available_duration
                changes.append(f"Shortened special note at {time:.3f}s to fit audio")

        updated_note = (
            note
            if time == note.time and note_duration == note.duration
            else ChartNote(
                time=time,
                value=note.value,
                duration=note_duration,
                balloon_hits=note.balloon_hits,
                beat_position=None,
                duration_beats=None,
            )
        )
        slot = round(time / slot_seconds)
        previous = time_slots.get(slot)
        if previous is not None:
            if note.value in (5, 7) and previous.value not in (5, 7):
                retained.remove(previous)
                time_slots[slot] = updated_note
                retained.append(updated_note)
                changes.append(f"Replaced conflicting note with special note at {time:.3f}s")
            else:
                changes.append(f"Removed duplicate note at {time:.3f}s")
            continue
        time_slots[slot] = updated_note
        retained.append(updated_note)

    retained.sort(key=lambda note: note.time)
    if minimum_interval > 0:
        interval_filtered: list[ChartNote] = []
        last_kept_time = float("-inf")
        for note in retained:
            if note.time - last_kept_time < minimum_interval - 1e-9:
                if note.value in (5, 7) and interval_filtered:
                    previous = interval_filtered[-1]
                    if previous.value in (5, 7):
                        changes.append(
                            f"Removed overlapping special note at {note.time:.3f}s"
                        )
                        continue
                    interval_filtered[-1] = note
                    last_kept_time = note.time
                    changes.append(
                        f"Replaced too-close note with special note at {note.time:.3f}s"
                    )
                    continue
                changes.append(f"Removed note violating minimum interval at {note.time:.3f}s")
                continue
            interval_filtered.append(note)
            last_kept_time = note.time
        retained = interval_filtered

    if maximum_density is not None and duration is not None:
        note_limit = floor(maximum_density * duration + 1e-9)
        while len(retained) > note_limit:
            removable = [
                (index, note)
                for index, note in enumerate(retained)
                if note.value not in (5, 7)
                and (index == 0 or retained[index - 1].value not in (5, 7))
                and (index + 1 == len(retained) or retained[index + 1].value not in (5, 7))
            ]
            if not removable:
                raise ValueError(
                    "cannot satisfy maximum_density without removing special notes"
                )
            remove_index, removed = min(
                removable,
                key=lambda item: (
                    _note_priority(item[1], chart.bpm),
                    item[1].time,
                ),
            )
            changes.append(f"Removed note to satisfy density limit at {removed.time:.3f}s")
            retained.pop(remove_index)

    repaired = TjaChart(
        title=chart.title,
        bpm=chart.bpm,
        course=chart.course,
        level=chart.level,
        notes=retained,
        offset=chart.offset,
        wave=chart.wave,
        events=list(chart.events),
    )
    remaining_errors = validate_chart(
        repaired,
        duration=duration,
        minimum_interval=minimum_interval,
        maximum_density=maximum_density,
    )
    if remaining_errors:
        raise ValueError("chart could not be fully repaired: " + "; ".join(remaining_errors))
    return RepairReport(chart=repaired, changes=tuple(changes))


def _note_priority(note: ChartNote, bpm: float) -> float:
    beat = note.time * bpm / 60.0
    beat_strength = 1.0 if abs(beat - round(beat)) < 1e-6 else 0.0
    large_note = 0.25 if note.value in (3, 4) else 0.0
    return beat_strength + large_note

import pytest

from taiko_ai.chart.models import ChartEvent, ChartNote, TjaChart
from taiko_ai.chart.repair import repair_chart
from taiko_ai.chart.validator import validate_chart


def chart_with_notes(notes: list[ChartNote]) -> TjaChart:
    return TjaChart(
        title="Repair",
        bpm=120.0,
        course="Oni",
        level=5,
        notes=notes,
    )


def test_repair_removes_notes_that_are_too_close_and_keeps_events() -> None:
    original = chart_with_notes(
        [
            ChartNote(time=0.0, value=1),
            ChartNote(time=0.125, value=2),
            ChartNote(time=0.5, value=3),
        ]
    )
    original.events.append(ChartEvent(time=0.25, command="SCROLL", value=1.5))

    report = repair_chart(original, minimum_interval=0.2)

    assert [note.time for note in report.chart.notes] == [0.0, 0.5]
    assert report.chart.events == original.events
    assert any("minimum interval" in change for change in report.changes)
    assert validate_chart(report.chart, minimum_interval=0.2) == []


def test_repair_density_prefers_stronger_beats_and_preserves_special_notes() -> None:
    chart = chart_with_notes(
        [
            ChartNote(time=0.0, value=1),
            ChartNote(time=0.25, value=1),
            ChartNote(time=0.5, value=2),
            ChartNote(time=0.75, value=1),
            ChartNote(time=2.0, value=5, duration=0.25),
            ChartNote(time=3.0, value=7, duration=0.25, balloon_hits=8),
        ]
    )

    report = repair_chart(
        chart,
        duration=4.0,
        maximum_density=1.25,
    )

    assert [note.time for note in report.chart.notes] == [0.0, 0.5, 0.75, 2.0, 3.0]
    assert [note.value for note in report.chart.notes if note.value in (5, 7)] == [5, 7]
    assert validate_chart(report.chart, duration=4.0, maximum_density=1.25) == []


def test_repair_trims_notes_and_special_durations_to_audio_bounds() -> None:
    chart = chart_with_notes(
        [
            ChartNote(time=0.0, value=5, duration=2.0),
            ChartNote(time=3.0, value=1),
        ]
    )

    report = repair_chart(chart, duration=1.0)

    assert len(report.chart.notes) == 1
    assert report.chart.notes[0].duration == 1.0
    assert any("Shortened special note" in change for change in report.changes)
    assert validate_chart(report.chart, duration=1.0) == []


def test_repair_removes_non_finite_and_unmatched_notes() -> None:
    chart = chart_with_notes(
        [
            ChartNote(time=float("nan"), value=1),
            ChartNote(time=0.0, value=8),
            ChartNote(time=0.5, value=1),
        ]
    )

    report = repair_chart(chart)

    assert [note.value for note in report.chart.notes] == [1]
    assert len(report.changes) == 2
    assert validate_chart(report.chart) == []


def test_repair_rejects_invalid_bpm_or_impossible_special_density() -> None:
    invalid_bpm = chart_with_notes([ChartNote(time=0.0, value=1)])
    invalid_bpm.bpm = 0.0
    with pytest.raises(ValueError, match="BPM"):
        repair_chart(invalid_bpm)

    special_only = chart_with_notes(
        [ChartNote(time=0.0, value=5, duration=1.0)]
    )
    with pytest.raises(ValueError, match="without removing special notes"):
        repair_chart(special_only, duration=1.0, maximum_density=0.5)

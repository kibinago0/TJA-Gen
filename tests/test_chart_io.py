import pytest

from taiko_ai.chart.models import ChartEvent, ChartNote, TjaChart
from taiko_ai.chart.tja_parser import parse_tja
from taiko_ai.chart.validator import validate_chart, validate_tja
from taiko_ai.export.tja_writer import render_tja


def test_writer_round_trip_preserves_grid_notes_and_metadata() -> None:
    chart = TjaChart(
        title="Round trip",
        bpm=120,
        course="Oni",
        level=5,
        offset=0.25,
        wave="song.ogg",
        notes=[
            ChartNote(time=0.0, value=1),
            ChartNote(time=0.25, value=2),
            ChartNote(time=0.5, value=3),
            ChartNote(time=0.75, value=4),
            ChartNote(time=2.0, value=1),
        ],
    )

    parsed = parse_tja(render_tja(chart))

    assert parsed.title == chart.title
    assert parsed.bpm == chart.bpm
    assert parsed.course == chart.course
    assert parsed.level == chart.level
    assert parsed.offset == chart.offset
    assert parsed.wave == chart.wave
    assert parsed.notes == chart.notes


def test_parser_selects_requested_course() -> None:
    text = """TITLE:Two charts
BPM:120
COURSE:Easy
LEVEL:2
#START
1000,
#END
COURSE:Oni
LEVEL:8
#START
0200,
#END
"""

    chart = parse_tja(text, course="Oni")

    assert chart.course == "Oni"
    assert chart.level == 8
    assert chart.notes == [ChartNote(time=0.5, value=2)]


def test_validator_reports_invalid_chart() -> None:
    chart = TjaChart(
        title="Invalid",
        bpm=0,
        course="Oni",
        level=11,
        notes=[ChartNote(time=-0.1, value=10)],
    )

    errors = validate_chart(chart)

    assert "BPM must be positive" in errors
    assert "LEVEL must be between 1 and 10" in errors
    assert "invalid note value: 10" in errors
    assert "note time must not be negative" in errors


def test_writer_round_trip_preserves_roll_and_balloon_metadata() -> None:
    chart = TjaChart(
        title="Special notes",
        bpm=120,
        course="Oni",
        level=7,
        notes=[
            ChartNote(time=0.0, value=5, duration=1.0),
            ChartNote(time=2.0, value=7, duration=0.75, balloon_hits=15),
        ],
    )

    rendered = render_tja(chart)
    parsed = parse_tja(rendered)

    assert "#BALLOON:15" in rendered
    assert [note.value for note in parsed.notes] == [5, 7]
    assert [note.duration for note in parsed.notes] == [1.0, 0.75]
    assert parsed.notes[0].balloon_hits is None
    assert parsed.notes[1].balloon_hits == 15


def test_writer_rejects_special_note_without_duration() -> None:
    chart = TjaChart(
        title="Invalid roll",
        bpm=120,
        course="Oni",
        level=5,
        notes=[ChartNote(time=0.0, value=5)],
    )

    with pytest.raises(ValueError, match="positive duration"):
        render_tja(chart)


def test_parser_reads_existing_balloon_header_and_roll_end_marker() -> None:
    text = """TITLE:Special notes
BPM:120
COURSE:Oni
LEVEL:7
#BALLOON:12
#START
5000000000000008,
7000000000000008,
#END
"""

    chart = parse_tja(text)

    assert [note.value for note in chart.notes] == [5, 7]
    assert chart.notes[0].duration == 1.875
    assert chart.notes[1].duration == 1.875
    assert chart.notes[1].balloon_hits == 12


def test_parser_ignores_comments_after_balloon_hit_counts() -> None:
    text = """TITLE:Balloon comment
BPM:120
COURSE:Oni
LEVEL:5
BALLOON:10 //verified
#START
7000000000000008,
#END
"""

    chart = parse_tja(text)

    assert chart.notes[0].balloon_hits == 10


def test_parser_tracks_gimmick_event_positions_and_timing() -> None:
    text = """TITLE:Gimmicks
BPM:120
COURSE:Oni
LEVEL:5
#START
1000,
#BPMCHANGE 240
0200,
#SCROLL 1.5
0010,
#DELAY 0.25
0001,
#END
"""

    chart = parse_tja(text)

    assert [(event.command, event.value, event.beat_position) for event in chart.events] == [
        ("BPMCHANGE", 240.0, 4.0),
        ("SCROLL", 1.5, 8.0),
        ("DELAY", 0.25, 12.0),
    ]
    assert [note.time for note in chart.notes] == [0.0, 2.25, 3.5, 5.0]


def test_gimmick_writer_round_trip_preserves_midmeasure_command_positions() -> None:
    chart = parse_tja(
        """TITLE:Gimmicks
BPM:120
COURSE:Oni
LEVEL:5
#START
1000,
#BPMCHANGE 240
0200,
#SCROLL 1.5
0010,
#DELAY 0.25
0001,
#END
"""
    )

    parsed = parse_tja(render_tja(chart))

    assert parsed.events == chart.events
    assert [note.beat_position for note in parsed.notes] == [
        note.beat_position for note in chart.notes
    ]
    assert [note.time for note in parsed.notes] == pytest.approx(
        [note.time for note in chart.notes]
    )


def test_midmeasure_bpmchange_keeps_notes_on_their_tja_grid_slots() -> None:
    text = """TITLE:Midmeasure change
BPM:120
COURSE:Oni
LEVEL:5
#START
1000
#BPMCHANGE 240
0200,
#END
"""

    chart = parse_tja(text)
    reparsed = parse_tja(render_tja(chart))

    assert chart.events[0].beat_position == 2.0
    assert chart.events[0].time == 1.0
    assert [note.time for note in chart.notes] == [0.0, 1.125]
    assert [note.beat_position for note in reparsed.notes] == pytest.approx(
        [0.0, 2.5]
    )
    assert [note.time for note in reparsed.notes] == pytest.approx(
        [note.time for note in chart.notes]
    )


@pytest.mark.parametrize(
    ("command", "value"),
    [("BPMCHANGE", 0.0), ("SCROLL", float("inf")), ("UNKNOWN", 1.0)],
)
def test_writer_rejects_invalid_gimmick_events(command, value) -> None:
    chart = TjaChart(
        title="Invalid event",
        bpm=120,
        course="Oni",
        level=5,
        notes=[ChartNote(time=0.0, value=1)],
        events=[ChartEvent(time=0.0, command=command, value=value)],
    )

    with pytest.raises(ValueError):
        render_tja(chart)


def test_validator_rejects_special_notes_without_required_metadata() -> None:
    chart = TjaChart(
        title="Invalid special notes",
        bpm=120,
        course="Oni",
        level=5,
        notes=[
            ChartNote(time=0.0, value=5),
            ChartNote(time=1.0, value=7, duration=0.5),
        ],
    )

    errors = validate_chart(chart)

    assert "roll and balloon notes must have positive duration" in errors
    assert "balloon must have a positive hit count" in errors


def test_validator_rejects_unmatched_end_marker() -> None:
    chart = TjaChart(
        title="Unmatched end",
        bpm=120,
        course="Oni",
        level=5,
        notes=[ChartNote(time=0.0, value=8)],
    )

    assert "unmatched roll/balloon end marker" in validate_chart(chart)


def test_validator_checks_density_interval_duration_and_event_order() -> None:
    chart = TjaChart(
        title="Invalid timing",
        bpm=120,
        course="Oni",
        level=5,
        notes=[
            ChartNote(time=0.0, value=1),
            ChartNote(time=0.1, value=2),
            ChartNote(time=1.0, value=3),
        ],
        events=[
            ChartEvent(time=0.5, command="SCROLL", value=1.0),
            ChartEvent(time=0.25, command="BPMCHANGE", value=180.0),
        ],
    )

    errors = validate_chart(
        chart,
        duration=1.5,
        minimum_interval=0.2,
        maximum_density=1.0,
    )

    assert "note interval too short" in errors
    assert "event times must be in ascending order" in errors
    assert "note density exceeds maximum_density" in errors


def test_validator_checks_finite_timing_and_negative_delay() -> None:
    chart = TjaChart(
        title="Non-finite values",
        bpm=float("nan"),
        course="Oni",
        level=5,
        offset=float("inf"),
        notes=[ChartNote(time=float("nan"), value=1)],
        events=[ChartEvent(time=0.0, command="DELAY", value=-0.25)],
    )

    errors = validate_chart(chart)

    assert "BPM must be positive" in errors
    assert "BPM must be finite" in errors
    assert "OFFSET must be finite" in errors
    assert "note time must be finite" in errors
    assert "negative #DELAY is not supported" in errors


def test_tja_validator_reports_syntax_errors_and_valid_chart() -> None:
    valid = """TITLE:Valid
BPM:120
COURSE:Oni
LEVEL:5
#START
1000,
#END
"""
    invalid = valid.replace("1000,", "10x0,")

    assert validate_tja(valid) == []
    assert validate_tja(invalid) == [
        "invalid TJA syntax: invalid character in TJA note data: 'x'"
    ]


def test_parser_counts_an_empty_measure_in_elapsed_time() -> None:
    chart = parse_tja(
        """TITLE:Empty measure
BPM:120
COURSE:Oni
LEVEL:5
#START
1000,
,
1000,
#END
"""
    )

    assert [note.time for note in chart.notes] == [0.0, 4.0]

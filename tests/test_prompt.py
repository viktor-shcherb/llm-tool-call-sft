import re
from car_sales.prompt import make_now_iso, shuffle_sys_prompt

TZ = "Europe/Zurich"


def test_make_now_iso_stable_known_date():
    date = "2024-03-10"
    session_id = "session-abc"
    v1 = make_now_iso(date, session_id, TZ)
    v2 = make_now_iso(date, session_id, TZ)
    assert v1 == v2  # deterministic

    # format check: YYYY-MM-DDTHH:MM:SS+HH:MM
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", v1)


def test_make_now_iso_stable_unknown_date():
    date = "unknown"
    session_id = "session-xyz"
    v1 = make_now_iso(date, session_id, TZ)
    v2 = make_now_iso(date, session_id, TZ)
    assert v1 == v2  # still deterministic even when random branch is used
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", v1)


def test_make_now_iso_dst_offset_changes():
    # Winter vs summer in Europe/Zurich should have different offsets (+01 vs +02 typically)
    winter = make_now_iso("2024-01-10", "sess-win", TZ)
    summer = make_now_iso("2024-07-10", "sess-sum", TZ)

    winter_off = winter[-6:]
    summer_off = summer[-6:]

    assert winter_off != summer_off
    assert winter_off in {"+01:00", "+02:00", "-01:00", "-02:00"}
    assert summer_off in {"+01:00", "+02:00", "-01:00", "-02:00"}


def test_shuffle_sys_prompt_deterministic_and_timestamp():
    base_text = (
        "# Section A\n"
        "- bullet1\n"
        "- bullet2\n"
        "line x\n"
        "# Section B\n"
        "- bullet3\n"
        "- bullet4\n"
    )
    now_iso = "2024-03-10T12:00:00+01:00"
    session_id = "sess-123"

    p1 = shuffle_sys_prompt(
        base_text=base_text,
        now_iso=now_iso,
        timezone_str=TZ,
        session_id=session_id,
    )
    p2 = shuffle_sys_prompt(
        base_text=base_text,
        now_iso=now_iso,
        timezone_str=TZ,
        session_id=session_id,
    )
    assert p1 == p2  # deterministic for a given session_id

    # timestamp line must be first
    first_line = p1.split("\n", 1)[0]
    assert first_line.startswith("Current DateTime (Europe/Zurich): ")
    assert now_iso in first_line

    # different session_id can change order of bullets / sections
    p3 = shuffle_sys_prompt(
        base_text=base_text,
        now_iso=now_iso,
        timezone_str=TZ,
        session_id="sess-999",
    )
    assert p3 != p1

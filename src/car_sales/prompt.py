import datetime
import hashlib
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

SECTION_REGEX = r"(?m)^# [^\n]+\n"


def _offset_str(dt: datetime.datetime) -> str:
    """
    Return timezone offset like '+01:00' for a timezone-aware datetime.
    """
    offset = dt.utcoffset()
    if offset is None:
        raise ValueError("datetime must be timezone-aware")

    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _seeded_rng(session_id: str) -> random.Random:
    """
    Return a Random() instance seeded from session_id.
    Stable across runs.
    """
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16)
    return random.Random(seed_int)


def make_now_iso(
    example_date: str,
    session_id: str,
    timezone_str: str,
) -> str:
    """
    Produce deterministic "current datetime" string for this session.

    Input
      example_date: "YYYY-MM-DD" or "unknown"
      session_id: unique session identifier
      timezone_str: e.g. "Europe/Zurich"

    Output
      "YYYY-MM-DDTHH:MM:SS+HH:MM"
      with correct local wall time and offset for that tz.
    """
    tz = ZoneInfo(timezone_str)

    if example_date and example_date != "unknown":
        # fixed noon at given date in that tz
        year, month, day = map(int, example_date.split("-"))
        local_dt = datetime.datetime(
            year,
            month,
            day,
            12,
            0,
            0,
            tzinfo=tz,
        )
    else:
        rng = _seeded_rng(session_id)

        year = rng.randint(2020, 2029)

        jan1 = datetime.date(year, 1, 1)
        dec31 = datetime.date(year, 12, 31)
        days_in_year = (dec31 - jan1).days + 1
        day_offset = rng.randrange(days_in_year)
        day_obj = jan1 + datetime.timedelta(days=day_offset)

        hour = rng.randrange(24)
        minute = rng.randrange(60)
        second = rng.randrange(60)

        local_dt = datetime.datetime(
            day_obj.year,
            day_obj.month,
            day_obj.day,
            hour,
            minute,
            second,
            tzinfo=tz,
        )

    off = _offset_str(local_dt)
    return local_dt.strftime("%Y-%m-%dT%H:%M:%S") + off



def load_sysprompt_text(sysprompt_path: str | Path) -> str:
    """
    Read base system prompt markdown from disk once.
    """
    p = Path(sysprompt_path)
    return p.read_text()


def load_global_tools(tools_path: str | Path) -> List[Dict[str, Any]]:
    """
    Read tool schema JSON into OpenAI/Qwen style tool descriptors.

    Input file format example:
    [
      {
        "function": {
          "name": "lookup_inventory",
          "description": "...",
          "parameters": { ... JSON schema ... }
        }
      },
      ...
    ]

    Output format:
    [
      {
        "type": "function",
        "function": {
          "name": <name>,
          "description": <description>,
          "parameters": <parameters>
        }
      },
      ...
    ]
    """
    p = Path(tools_path)
    raw_tools = json.loads(p.read_text())

    tools: List[Dict[str, Any]] = []
    for t in raw_tools:
        func = t.get("function", {})
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get(
                        "description",
                        f"Internal function {func['name']}.",
                    ),
                    "parameters": func.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    ),
                },
            }
        )
    return tools


def shuffle_sys_prompt(
    base_text: str,
    now_iso: str,
    timezone_str: str,
    session_id: str,
) -> str:
    """
    Deterministically randomize system prompt structure for a session.

    Steps:
    - split base_text into sections by "^# ...\n"
    - within each section shuffle bullet lines ("- ")
    - shuffle section order
    - prepend timestamp line
    All shuffles use a RNG seeded by session_id.
    """
    rng = _seeded_rng(session_id)

    parts = re.split(SECTION_REGEX, base_text)
    headers = re.findall(SECTION_REGEX, base_text)

    if not headers:
        sections = [base_text]
    else:
        sections: List[str] = []

        for h, b in zip(headers, parts[1:]):
            body_lines = b.strip("\n").split("\n")

            bullets = [ln for ln in body_lines if ln.lstrip().startswith("- ")]
            non_bullets = [ln for ln in body_lines if not ln.lstrip().startswith("- ")]

            if bullets:
                rng.shuffle(bullets)
                body_lines_shuffled = non_bullets + bullets
            else:
                body_lines_shuffled = body_lines

            section_text = h + "\n".join(body_lines_shuffled).strip() + "\n"
            sections.append(section_text)

        rng.shuffle(sections)

        if parts[0].strip():
            sections.insert(0, parts[0].strip() + "\n")

    preface = f"Current DateTime ({timezone_str}): {now_iso}\n"

    return preface + "\n".join(sections).strip()


def session_to_messages(
    example: Dict[str, Any],
    base_sysprompt_text: str,
    timezone_str: str,
) -> List[Dict[str, Any]]:
    """
    Convert dataset row into chat messages for training/inference.
    """
    now_iso = make_now_iso(
        example_date=example["inferred_date"],
        session_id=example["session_id"],
        timezone_str=timezone_str,
    )

    sys_prompt = shuffle_sys_prompt(
        base_text=base_sysprompt_text,
        now_iso=now_iso,
        timezone_str=timezone_str,
        session_id=example["session_id"],
    )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": sys_prompt,
        }
    ]

    for turn in example["conversation"]:
        role = turn["role"]
        content = turn.get("content", "") or ""

        if role == "user":
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )

        elif role == "assistant":
            tcalls = turn.get("tool_calls")
            if tcalls:
                calls_struct = []
                for call in tcalls:
                    calls_struct.append(
                        {
                            "id": call["id"],
                            "type": call.get("type", "function"),
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": json.loads(
                                    call["function"]["arguments"]
                                ) if isinstance(call["function"]["arguments"], str)
                                else call["function"]["arguments"]
                            },
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": calls_struct,
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.get("tool_call_id", ""),
                    "content": content,
                }
            )

    return messages


__all__ = [
    "_offset_str",
    "_seeded_rng",
    "make_now_iso",
    "shuffle_sys_prompt",
    "load_sysprompt_text",
    "load_global_tools",
    "session_to_messages",
]

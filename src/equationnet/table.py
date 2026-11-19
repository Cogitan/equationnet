"""Reading a spreadsheet's CSV export for the guided session.

Stricter readers (fit()'s, the saved scripts') stop at the first cell that is not a
number. Exported spreadsheets often have a notes column or clock times, so here:
  - a column that is mostly text is not used, and the session says so;
  - a column of clock times or dates becomes hours since its first row, and the
    session says so;
  - a stray cell that is not a number, in a column of numbers, counts as missing, and
    the session names it (its row is left out of the fit, which also says so).
Nothing is changed or dropped without a line saying what.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import numpy as np

MISSING = {"", "nan", "na", "n/a", "null", "none", "-"}
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?\s*([AaPp][Mm])?$")


class Columns(dict):
    """Column name -> values (floats, NaN for missing), plus notes in words about
    anything that was read in a way the user should know."""

    def __init__(self, *a, notes=None, skipped=None, **k):
        super().__init__(*a, **k)
        self.notes = list(notes or [])
        self.skipped = dict(skipped or {})      # column name -> why it is not used


def _num(cell: str):
    c = cell.strip().replace("−", "-")     # a typographic minus from some exports
    if c.lower() in MISSING:
        return np.nan
    try:
        return float(c)
    except ValueError:
        return None


def _when(cell: str):
    """Seconds since midnight for a clock time; a datetime for an ISO date or date-time;
    None otherwise."""
    c = cell.strip()
    m = _CLOCK.match(c)
    if m:
        h, mi, s, ampm = int(m.group(1)), int(m.group(2)), float(m.group(3) or 0), m.group(4)
        if ampm:
            if not 1 <= h <= 12:
                return None
            h = h % 12 + (12 if ampm.lower() == "pm" else 0)
        if h > 23 or mi > 59 or s >= 60:
            return None
        return h * 3600 + mi * 60 + s
    try:
        return datetime.fromisoformat(c.replace("Z", "+00:00"))
    except ValueError:
        return None


def _times_to_hours(cells):
    """Hours since the first time in the column, or None if the cells are not all times
    of one kind. Clock times that go backwards are taken to have passed midnight."""
    vals = [None if c.strip().lower() in MISSING else _when(c) for c in cells]
    present = [v for v in vals if v is not None]
    raw = [c for c in cells if c.strip().lower() not in MISSING]
    if not present or len(present) != len(raw):
        return None, None
    if all(isinstance(v, float) for v in present):
        out, day, last = [], 0.0, None
        for v in vals:
            if v is None:
                out.append(np.nan)
                continue
            if last is not None and v + day < last - 1e-9:
                day += 86400.0
            out.append(v + day)
            last = v + day
        arr = np.array(out)
        kind = "clock times"
    elif all(isinstance(v, datetime) for v in present):
        aware = {v.tzinfo is not None for v in present}
        if len(aware) > 1:
            return None, None
        t0 = next(v for v in vals if v is not None)
        arr = np.array([np.nan if v is None else (v - t0).total_seconds() for v in vals])
        kind = "dates"
    else:
        return None, None
    first = arr[np.isfinite(arr)][0]
    return (arr - first) / 3600.0, kind


def read_table(path) -> Columns:
    """Read a CSV for the guided session. Raises ValueError or OSError with a message
    in words when the file cannot be used at all."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path.name} is empty")
    head = rows[0]
    if len(head) == 1:
        for sep, word in ((";", "semicolons"), ("\t", "tabs"), ("|", "'|'")):
            if sep in head[0]:
                raise ValueError(f"{path.name} seems to separate values with {word}; "
                                 "equationnet reads comma-separated files (save as CSV with commas)")
    body = rows[1:]
    if not body:
        raise ValueError(f"{path.name} has a header but no data rows")
    width = max(len(head), *(len(r) for r in body))
    names = [h.strip() for h in head] + [""] * (width - len(head))
    body = [r + [""] * (width - len(r)) for r in body]
    cols, notes, skipped = {}, [], {}
    seen = {}
    for j, name in enumerate(names):
        cells = [r[j] for r in body]
        if not name:
            if any(c.strip() for c in cells):
                name = f"column {j + 1}"
                notes.append(f"column {j + 1} has no name in the header; it is called '{name}' here")
            else:
                continue                        # an empty unnamed column (Excel adds these)
        if name in seen:
            seen[name] += 1
            new = f"{name} ({seen[name]})"
            notes.append(f"the header has '{name}' twice; the second is called '{new}' here")
            name = new
        else:
            seen[name] = 1
        nums = [_num(c) for c in cells]
        bad = [(i + 1, c.strip()) for i, (c, v) in enumerate(zip(cells, nums)) if v is None]
        filled = sum(1 for c in cells if c.strip().lower() not in MISSING)
        if not filled:
            skipped[name] = "empty"
            continue
        if not bad:
            cols[name] = np.array(nums, float)
            continue
        hours, kind = _times_to_hours(cells)
        if hours is not None:
            cols[name] = hours
            notes.append(f"'{name}' holds {kind}; read as hours since its first row "
                         f"(0 to {np.nanmax(hours):.4g} h)")
            continue
        if len(bad) * 2 > filled:               # mostly text: a notes or label column
            skipped[name] = f"text, e.g. {bad[0][1]!r} in data row {bad[0][0]}"
            continue
        shown = ", ".join(f"row {i} {c!r}" for i, c in bad[:3]) + (", ..." if len(bad) > 3 else "")
        notes.append(f"'{name}': {len(bad)} cell{'s are' if len(bad) > 1 else ' is'} not a number "
                     f"({shown}); counted as missing, so {'those rows are' if len(bad) > 1 else 'that row is'} "
                     f"left out if this column is used")
        cols[name] = np.array([np.nan if v is None else v for v in nums], float)
    return Columns(cols, notes=notes, skipped=skipped)

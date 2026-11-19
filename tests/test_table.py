"""The guided session's CSV reader (equationnet.table): spreadsheet exports with notes
columns, clock times and stray text are read, and every such reading is said."""

import numpy as np
import pytest

from equationnet.table import read_table


def write(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_numbers_only_unchanged(tmp_path):
    c = read_table(write(tmp_path, "t,u\n0,1\n1,2.5\n2,-3e-2\n"))
    assert list(c) == ["t", "u"] and np.allclose(c["u"], [1, 2.5, -0.03])
    assert c.notes == [] and c.skipped == {}


def test_notes_column_is_not_used_and_said(tmp_path):
    c = read_table(write(tmp_path, "min,temp,Notes\n0,190,poured\n3,177,\n6,169,\n9,158,lid off\n"))
    assert list(c) == ["min", "temp"]
    assert "Notes" in c.skipped and "poured" in c.skipped["Notes"]


def test_stray_text_counts_as_missing_and_is_named(tmp_path):
    c = read_table(write(tmp_path, "t,u\n0,1\n1,oops\n2,3\n3,4\n"))
    assert np.isnan(c["u"][1]) and c["u"][2] == 3
    assert any("oops" in n and "row 2" in n for n in c.notes)


@pytest.mark.parametrize("cells, hours", [
    (["10:00", "10:30", "11:15"], [0, 0.5, 1.25]),
    (["23:30", "00:15", "01:00"], [0, 0.75, 1.5]),                 # past midnight
    (["9:00 AM", "12:30 PM", "1:00 PM"], [0, 3.5, 4]),
    (["2025-01-01", "2025-01-02", "2025-01-04"], [0, 24, 72]),
    (["2025-01-01 10:00", "2025-01-01T10:30:00", "2025-01-01 12:00"], [0, 0.5, 2]),
])
def test_clock_times_and_dates_become_hours_and_are_said(tmp_path, cells, hours):
    text = "when,u\n" + "".join(f"{c},{i}\n" for i, c in enumerate(cells))
    c = read_table(write(tmp_path, text))
    assert np.allclose(c["when"], hours)
    assert any("'when'" in n and "hours since its first row" in n for n in c.notes)


def test_header_problems_are_named_not_fatal(tmp_path):
    c = read_table(write(tmp_path, "t,u,u,\n0,1,2,5\n1,2,3,6\n"))
    assert list(c) == ["t", "u", "u (2)", "column 4"]
    assert len(c.notes) == 2


def test_empty_unnamed_columns_vanish(tmp_path):
    c = read_table(write(tmp_path, "t,u,,\n0,1,,\n1,2,,\n"))
    assert list(c) == ["t", "u"] and c.notes == []


@pytest.mark.parametrize("text, words", [
    ("", "is empty"),
    ("t,u\n", "no data rows"),
    ("t;u\n1;2\n", "semicolons"),
])
def test_unusable_files_say_why(tmp_path, text, words):
    with pytest.raises(ValueError, match=words):
        read_table(write(tmp_path, text))

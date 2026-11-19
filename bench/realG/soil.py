"""Stage G, PDE: soil temperature from NOAA's US Climate Reference Network (USCRN).

    python bench/realG/soil.py HOURLY_FILE HEADERS_FILE [--out bench/realG/soil_lincoln_2025.csv]

HOURLY_FILE is the station-year file from
https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2025/CRNH0203-2025-NE_Lincoln_11_SW.txt (Austin 33 NW was the first choice; it has no sensors below 10 cm)
and HEADERS_FILE is https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/HEADERS.txt
(US government data, public domain; credit: NOAA National Centers for Environmental
Information, U.S. Climate Reference Network). Only the derived table is kept here.

Window rule (TESTPLAN.md, stage G PDE, written before the file was opened): 7 days
(168 hours) from 2025-04-01 00:00 local standard time; if any of the five soil
temperatures is missing in it, the first later 7-day window starting at local
midnight with none missing. Each hourly value is placed at the end of its hour.
Output columns: x (depth, m), t (days from the window start), u (degrees C).
"""

import argparse
import csv
from datetime import datetime, timedelta

DEPTHS = {"SOIL_TEMP_5": 0.05, "SOIL_TEMP_10": 0.10, "SOIL_TEMP_20": 0.20,
          "SOIL_TEMP_50": 0.50, "SOIL_TEMP_100": 1.00}
MISSING = -9999.0
DAYS = 7
FIRST = datetime(2025, 4, 1)


def read(hourly, headers):
    names = open(headers).read().split("\n")[1].split()
    rows = {}
    for line in open(hourly):
        f = dict(zip(names, line.split()))
        if not f:
            continue
        d, hhmm = f["LST_DATE"], int(f["LST_TIME"])
        when = datetime.strptime(d, "%Y%m%d") + timedelta(hours=hhmm // 100, minutes=hhmm % 100)
        rows[when] = {k: float(f[k]) for k in DEPTHS}
    return rows


def window(rows):
    start = FIRST
    while start.year == FIRST.year:
        hours = [start + timedelta(hours=h + 1) for h in range(24 * DAYS)]
        if all(h in rows and all(v != MISSING for v in rows[h].values()) for h in hours):
            return start, hours
        start += timedelta(days=1)
    raise SystemExit("no complete 7-day window in the file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hourly")
    ap.add_argument("headers")
    ap.add_argument("--out", default="bench/realG/soil_lincoln_2025.csv")
    a = ap.parse_args()
    rows = read(a.hourly, a.headers)
    start, hours = window(rows)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "t", "u"])
        for h in hours:
            for k, depth in DEPTHS.items():
                w.writerow([depth, round((h - start).total_seconds() / 86400, 6), rows[h][k]])
    print(f"window from {start} local standard time, {len(hours)} hours x {len(DEPTHS)} depths -> {a.out}")


if __name__ == "__main__":
    main()

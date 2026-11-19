"""Stage G, PDE (diagnosis after the run): diffusivity read off the daily cycle.

    python bench/realG/soil_harmonics.py

At each depth, fit u = c0 + c1*t + c2*t^2 + A*cos(w*t - phi) with w = 2*pi per day
(t in days), by least squares on the week (and on each half of it). For a uniform
layer under a steady daily sine, the amplitude falls as exp(-dz/d) and the phase
lags by dz/d, with d = sqrt(2*a/w); so between two depths

    a from amplitude = w*dz^2 / (2*ln(A1/A2)^2)
    a from phase     = w*dz^2 / (2*(phi2 - phi1)^2)

Real weather is not a steady sine and the trend terms are crude, so these are rough.
Also prints the held top-edge value and its error on the forecast split.
"""

from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = 2 * np.pi


def harmonics(t, u):
    A = np.column_stack([np.ones_like(t), t, t ** 2, np.cos(W * t), np.sin(W * t)])
    c = np.linalg.lstsq(A, u, rcond=None)[0]
    return np.hypot(c[3], c[4]), np.arctan2(c[4], c[3]), u - A @ c


def estimates(d, keep):
    depths = np.unique(d["x"])
    h = {z: harmonics(d["t"][keep & np.isclose(d["x"], z)], d["u"][keep & np.isclose(d["x"], z)])
         for z in depths}
    out = {}
    for z1, z2 in combinations(depths, 2):
        (a1, p1, _), (a2, p2, _) = h[z1], h[z2]
        dz = z2 - z1
        out[(z1, z2)] = (W * dz ** 2 / (2 * np.log(a1 / a2) ** 2),
                         W * dz ** 2 / (2 * ((p2 - p1) % (2 * np.pi)) ** 2))
    return h, out


def main():
    d = np.genfromtxt(HERE / "soil_lincoln_2025.csv", delimiter=",", names=True)
    halves = {"whole week": np.ones(len(d), bool), "days 0-3.5": d["t"] <= 3.5,
              "days 3.5-7": d["t"] > 3.5}
    for name, keep in halves.items():
        h, out = estimates(d, keep)
        print(f"\n{name}: daily amplitude (C) and phase (rad) by depth")
        for z, (amp, ph, res) in h.items():
            lag1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
            print(f"  {z:5.2f} m  {amp:6.3f}  {ph:7.3f}   hourly residual lag-1 correlation {lag1:.2f}")
        print("  depths (m)     a from amplitude   a from phase   (m^2/day)")
        for (z1, z2), (aa, ap) in out.items():
            print(f"  {z1:.2f}-{z2:.2f}     {aa:10.4f}       {ap:10.4f}")
    # forecast split: the top edge is held at its value at the last training time
    ut = np.unique(d["t"])
    k = int(round(0.2 * len(ut)))
    last = ut[-k - 1]
    top = np.isclose(d["x"], 0.05)
    held = float(d["u"][top & np.isclose(d["t"], last)][0])
    after = top & (d["t"] > last)
    rms = float(np.sqrt(np.mean((d["u"][after] - held) ** 2)))
    print(f"\nforecast: last training time t = {last:.4f} days ({(last % 1) * 24:.0f}:00 local), "
          f"top edge held at {held} C; RMS of that held value against the 5 cm data after it: "
          f"{rms:.2f} C over {after.sum()} hours")
    sd10 = float(np.std(d["u"][np.isclose(d["x"], 0.10) & (d["t"] > last)]))
    print(f"spread (SD) of the 10 cm data in the held-out time: {sd10:.2f} C")


if __name__ == "__main__":
    main()

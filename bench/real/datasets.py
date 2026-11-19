"""Real measured data with a known governing equation, for testing equationnet.

All values are as listed in the data sets of Lomen and Lovelock, "Differential
Equations: Graphics, Models, Data" (Wiley, 1999), published at
https://archive.math.arizona.edu/dsl/datall.htm (read 2026-11-16). The original
source of each set is given below. Each entry: columns, equations to try, and what
is known independently of the data (to judge the fitted constants).

These are third-party measurements, not covered by the repository's MIT license;
they are reproduced with credit to the sources named here.
"""

import numpy as np

DATASETS = {
    "yeast": dict(
        source='T. Carlson, "Growth of Yeast", Biochemische Zeitschrift 57 (1913) 313',
        page="https://archive.math.arizona.edu/dsl/byeast.htm",
        t=np.arange(19.0),   # hours
        u=np.array([9.6, 18.3, 29.0, 47.2, 71.1, 119.1, 174.6, 257.3, 350.7, 441.0, 513.3,
                    559.7, 594.8, 629.4, 640.8, 651.1, 655.9, 659.6, 661.8]),  # amount of yeast
        equations=["u_t = r*u*(1 - u/K)"],
        known="classic logistic data; published logistic fits give K about 665, r about 0.53/h",
    ),
    "coffee": dict(
        source="students' measurement in a room at 24 C (Lomen and Lovelock, chapter 4)",
        page="https://archive.math.arizona.edu/dsl/bcoffee.htm",
        t=np.arange(11.0),   # minutes
        u=np.array([82.3, 79.7, 77.4, 75.1, 73.2, 70.8, 69.3, 67.0, 65.4, 63.4, 62.1]),  # deg C
        equations=["u_t = -k*(u - A)", "u_t = -k*(u - 24)"],
        known="room temperature A = 24 C (stated with the data)",
    ),
    "rlc_series": dict(
        source="series RLC circuit, R = 2500 ohm, L = 0.6 H, C = 3.5e-6 F; experiment by a "
               "University of Arizona student (Lomen and Lovelock, chapter 8)",
        page="https://archive.math.arizona.edu/dsl/brcl-s.htm",
        t=np.array([0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]),  # s
        u=np.array([0.200, 0.100, 0.060, 0.033, 0.020, 0.011, 0.007, 0.004, 0.002]),  # volts
        equations=["u_tt + c*u_t + k*u = 0"],
        known="from the components: c = R/L = 4167 /s, k = 1/(L*C) = 476,190 /s^2 "
              "(overdamped: decay rates 118 and 4049 /s)",
    ),
    "overdamped_pendulum": dict(
        source='Greenwood et al., "Using videotapes to study damped harmonic motion and to '
               'measure terminal speeds", Am. J. Phys. 54 (1986) 897, Figure 4',
        page="https://archive.math.arizona.edu/dsl/boverdam.htm",
        t=np.array([0.920, 0.989, 1.058, 1.121, 1.191, 1.254, 1.317, 1.386, 1.449, 1.518,
                    1.581, 1.650, 1.713, 1.789, 1.858, 1.928, 1.984, 2.054, 2.117, 2.192,
                    2.255, 2.318, 2.387, 2.457, 2.520, 2.589, 2.652]),  # s
        u=np.array([5.182, 4.827, 4.364, 3.982, 3.682, 3.491, 3.055, 2.864, 2.591, 2.427,
                     2.236, 2.045, 1.909, 1.773, 1.636, 1.582, 1.364, 1.309, 1.145, 1.091,
                     1.036, 0.982, 0.927, 0.845, 0.791, 0.764, 0.736]),  # cm
        equations=["u_tt + c*u_t + k*u = 0"],
        known="overdamped (no oscillation); constants not given",
    ),
    "theophylline": dict(
        source='Burghes, Huntley and McDonald, "Applying Mathematics" (Ellis Horwood, 1982) '
               "p. 125: blood concentration after a 300 mg injection",
        page="https://archive.math.arizona.edu/dsl/bdrugs.htm",
        t=np.array([0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19.0]),  # hours
        u=np.array([12.0, 10.0, 7.0, 5.0, 3.5, 2.5, 2.0, 1.5, 1.0, 0.7, 0.5]),  # mg/l
        equations=["u_t = -k*u"],
        known="one-compartment elimination assumed; no independent value checked",
    ),
    "calf_dye": dict(
        source='Harrison et al., "The Measurement of Liver Blood Flow in Conscious Calves", '
               "Q. J. Exp. Physiol. 74 (1988) 235",
        page="https://archive.math.arizona.edu/dsl/bcalves.htm",
        t=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19, 22, 25, 28, 31.0]),  # minutes
        u=np.array([10.20, 7.67, 5.76, 4.50, 3.56, 2.77, 2.30, 1.84, 1.46, 1.26, 0.77,
                    0.52, 0.39, 0.28, 0.22, 0.17, 0.14]),  # mg/ml x 100
        equations=["u_t = -k*u", "u_t = -f(u)"],
        known="dye clearance is usually two-compartment (two exponentials), so u_t = -k*u "
              "is expected to be wrong; f(u) lets the data choose the decay law",
    ),
}

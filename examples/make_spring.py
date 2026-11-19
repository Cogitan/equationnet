"""Writes spring.csv, the README's example data: 40 noisy points of a damped spring
u'' + 0.3 u' + 4 u = 0, u(0) = 1, u'(0) = 0, over t = 0 to 10, noise 0.05 (seed 1)."""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

t = np.linspace(0, 10, 40)
u = solve_ivp(lambda tt, y: [y[1], -0.3 * y[1] - 4 * y[0]], (0, 10), [1, 0], t_eval=t,
              rtol=1e-10, atol=1e-12).y[0]
u = u + 0.05 * np.random.default_rng(1).standard_normal(40)
lines = ["t,u"] + [f"{a:.6g},{b:.6g}" for a, b in zip(t, u)]
(Path(__file__).parent / "spring.csv").write_text("\n".join(lines) + "\n")

# Stage I7 review record (2026-11-19)

The reviewer was an AI agent (Anthropic's Claude, the same model family that wrote the
code), started without the code, the README or the author's notes. It was told to play
a food-science graduate student who had never used the tool and was given only: "pip
install equationnet, then type equationnet", the installed command, and the CSV
`coffee_cooling.csv` (31 rows; `Minutes elapsed`, `Coffee temp (F)`, `Room temp (F)`,
`Notes`; made with L = 72 F, k = 0.035 per minute, noise 0.8 F). It was told not to open
the README or the package source. Its shell could not hold an interactive session, so it
piped in the answers decided so far and re-ran, one answer more each time (numbered
menus; the arrow-key menus were out of its scope). It ran the code as it was before the
fixes listed in TESTPLAN.md, Stage I results.

Its report, condensed by the author of the code (on-screen quotes kept exactly; local paths
shortened to `<review>`; "my thermometer" written as the room-temperature column it read):

---

Ran as the persona and got a result: the Newton's-cooling preset worked, and predict and
save-script worked. Four real problems: a silent misreading of a mistyped equation, the
"Careful" option crashing on the setting the program had recommended, a menu wording that
steers toward the wrong equation, and two unexplained sets of constants. About 12 runs.

2(a) Did the equation help? On the first run, yes, about 1.5 times better than ignoring it:
"Your equation helped. ... the best model using it (the classical fit) was off by 0.6306
... and the best data-only model (the neural network) by 0.9145, 1.5 times as much."
The "Careful" re-run (scattered points) says it can't tell: "This data cannot tell whether
your equation helps. ... (the classical fit, off by 0.6398) and the best data-only model
(the polynomial, off by 0.7003) are within the noise of each other". So: it helps at
predicting what comes next, and is no better at filling gaps. The program never puts those
two verdicts side by side.

2(b) The constants. First run: L=72.76+-0.66, k=0.03569+-0.00056. Careful run, and also
the saved script: L=72.29+-0.41, k=0.03537+-0.0004. That is a room temperature of about
72.3 +- 0.4 F (the room-temperature column reads 71.4 to 72.4) and a cooling constant of
about 0.0354 +- 0.0004. The unit of k is never stated; I assume per minute.

2(c) Trust: fairly high. L matches the room-temperature column, and the wrong equation
(steady decay) clearly lost (held-out error 17.15). But the +- only holds "if the equation
is exactly right", and the notes say "lid off" and "stirred" partway through. The
120-minute prediction came back as 120 74.3838, with no uncertainty and no warning that
120 is past the last point at 90.

Where I was confused, hesitated, or would have given up:
1. Wrong default column: number [3] defaults to "Room temp (F)" for "Which column did you
   measure?".
2. Room temp offered as "position": "Does Coffee temp (F) also change with position (a
   second variable)? ... 2. Yes: Room temp (F)".
3. The room-temp column is thrown away: "not used: Room temp (F)", never offered as L.
4. The menu steers toward the wrong equation: "Steady decay" is described as "cooling
   toward zero"; "Relaxing toward a level" lists "warming up, charging, oxygen demand".
5. Silent misreading of my equation (worst): `u_t = k(L-u)` gave only "Constants to fit:
   L", treating k as a function to learn; found only after training, from "learn k(L - u)"
   and "L=42.23 (+-inf)".
6. Unhelpful error for a wrong letter: `T_t = -k*(T - 72)` gave "the equation has no
   derivative of u".
7. "Careful" crashes on the advised setting: with the forecast option it failed with
   "That did not run: cv=5 forecast origins of 6 time(s) each need more than 31 distinct
   times; there are 31. Use fewer folds or a smaller holdout", and sent me back to the
   equation menu.
8. Two different constant sets, no explanation: the script printed L 72.292956, k
   0.035369, not the screen's 72.76 / 0.03569.
9. Absurd constants with no warning: two equation models reported "L=2.149" and
   "L=0.755" with error 14.6.
10. Coding instructions for a non-programmer: "Run one from that folder with: python
    equation_fit.py ...".

Wrong, misleading or jargon: the wrong equation still prints a tight-looking constant
("k=0.01275+-0.00057"); the "not pinned down" note says "pass known={'L': ...}", which is
Python code; terms a non-expert won't know: "PINN", "RMS", "cv=5 forecast origins",
"folds", "holdout", "one standard error", "the one the equation constrains most",
"within the noise".

Answer sequences piped (main run): 1, ../coffee_cooling.csv, 2, 1, 1, 2, n, 2, 1, 4,
two hours, 120, n, 2, (empty). Saved script run from equationnet_output:
`python equation_fit.py coffee_cooling_equationnet.csv`. Careful run: 1,
../coffee_cooling.csv, 2, 1, 1, 2, n, 1, 2. Different equation: 1, ../coffee_cooling.csv,
2, 1, 1, 1, n, 2, 1. Mistyped equation: as the main run through Standard, then 1, 6,
`u_t = k(L-u)`, n, 2, 1. Outputs were kept in `<review>/run` (not in this repository).

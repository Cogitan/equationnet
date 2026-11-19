"""Stage G real data: five NIST Statistical Reference Datasets (nonlinear regression).

Data and certified values as published by NIST (third-party data, not covered by the
repository's MIT license; reproduced with credit), https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml, read 2026-11-17.
Each page: .../strd/nls/data/LINKS/DATA/<name>.dat. The original source of each set is
given below. "certified" = NIST certified least-squares value and standard deviation
for the closed-form model; the equations here are that model written as an ODE (same
constants), with the start value left free.
"""

import numpy as np

DATASETS = {
    "Misra1a": dict(
        source="volume vs pressure, NIST dental monomolecular adsorption study (Misra 1978)",
        page="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat",
        var="x",
        x=np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0]),
        u=np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.4, 75.47, 81.78]),
        equations=['u_x = k*(A - u)', 'u_x = k*u'],
        certified={'A': (238.94212918, 2.7070075241), 'k': (0.00055015643181, 7.2668688436e-06)},
    ),
    "Rat42": dict(
        source="pasture yield vs growing time (Ratkowsky 1983)",
        page="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Rat42.dat",
        var="t",
        x=np.array([9.0, 14.0, 21.0, 28.0, 42.0, 57.0, 63.0, 70.0, 79.0]),
        u=np.array([8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08]),
        equations=['u_t = r*u*(1 - u/K)'],
        certified={'K': (72.462237576, 1.7340283401), 'r': (0.067359200066, 0.0034465663377)},
    ),
    "Eckerle4": dict(
        source="transmittance vs wavelength, NIST circular interference transmittance study (Eckerle)",
        page="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Eckerle4.dat",
        var="x",
        x=np.array([400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 444.0, 445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5, 456.0, 457.5, 459.0, 460.5, 462.0, 463.5, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0]),
        u=np.array([0.0001575, 0.0001699, 0.000235, 0.0003102, 0.0004917, 0.000871, 0.0017418, 0.00464, 0.0065895, 0.0097302, 0.0149002, 0.023731, 0.0401683, 0.0712559, 0.1264458, 0.2073413, 0.2902366, 0.3445623, 0.3698049, 0.3668534, 0.3106727, 0.2078154, 0.1164354, 0.0616764, 0.03372, 0.0194023, 0.0117831, 0.0074357, 0.0022732, 0.00088, 0.0004579, 0.0002345, 0.0001586, 0.0001143, 7.1e-05]),
        equations=['u_x = -(x - m)*u/s^2'],
        certified={'m': (451.54121844, 0.046800518816), 's': (4.0888321754, 0.046803020753)},
    ),
    "BoxBOD": dict(
        source="biochemical oxygen demand (mg/l) vs incubation time in days (Box, Hunter and Hunter 1978)",
        page="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/BoxBOD.dat",
        var="t",
        x=np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
        u=np.array([109.0, 149.0, 149.0, 191.0, 213.0, 224.0]),
        equations=['u_t = k*(L - u)'],
        certified={'L': (213.80940889, 12.354515176), 'k': (0.54723748542, 0.10455993237)},
    ),
    "DanWood": dict(
        source="energy radiated by a carbon filament lamp vs temperature in 1000 K (Daniel and Wood 1980)",
        page="https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/DanWood.dat",
        var="x",
        x=np.array([1.309, 1.471, 1.49, 1.565, 1.611, 1.68]),
        u=np.array([2.138, 3.421, 3.597, 4.34, 4.882, 5.66]),
        equations=['u_x = b*u/x'],
        certified={'b': (3.8604055871, 0.051726610913)},
    ),
}

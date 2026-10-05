"""Generate SYNTHETIC CFRP impact-test data.

This data is simulated, NOT experimental. Because we know the true
parameters used to create it, we can later check that the analysis recovers
them, which validates the method.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.damage_models import delamination_area, residual_strength

# True parameters (illustrative values, not from a real material)
TRUE = {
    "E_th": 5.0,       # threshold energy (J)
    "c": 25.0,         # damage growth (mm^2 per J)
    "sigma0": 450.0,   # undamaged strength (MPa)
    "beta": 0.004,     # strength loss rate (1/mm^2)
    "r": 0.45,         # retained strength fraction at large damage
}
AREA_NOISE_FRAC = 0.10     # 10% scatter on damage area
STRENGTH_NOISE_SD = 12.0   # scatter on strength (MPa)
SEED = 42                  # fixed seed -> reproducible


def make_dataset(repeats=4, seed=SEED):
    """Several simulated specimens at each impact energy."""
    rng = np.random.default_rng(seed)
    levels = np.array([0, 3, 6, 9, 12, 16, 20, 25, 30, 40], dtype=float)  # J
    E = np.repeat(levels, repeats)

    A_true = delamination_area(E, TRUE["E_th"], TRUE["c"])
    A = A_true * (1.0 + rng.normal(0.0, AREA_NOISE_FRAC, size=E.shape))
    A = np.clip(A, 0.0, None)

    S = residual_strength(A, TRUE["sigma0"], TRUE["beta"], TRUE["r"])
    S = S + rng.normal(0.0, STRENGTH_NOISE_SD, size=E.shape)

    return pd.DataFrame({
        "impact_energy_J": E,
        "delam_area_mm2": A.round(1),
        "CAI_strength_MPa": S.round(1),
        "data_type": "synthetic",
    })


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "synthetic_cfrp.csv"
    out.parent.mkdir(exist_ok=True)
    df = make_dataset()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
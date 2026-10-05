import numpy as np
from src.damage_models import predict_strength_from_energy
from src.generate_data import make_dataset
from src.uncertainty import (
    bootstrap_chain, confidence_band, energy_for_strength_fraction,
)

TRUE_PARAMS = [5.0, 25.0, 450.0, 0.004, 0.45]


def test_energy_for_strength_fraction_round_trip():
    E = energy_for_strength_fraction(TRUE_PARAMS, 0.7)
    strength = predict_strength_from_energy(E, *TRUE_PARAMS)
    assert np.isclose(strength, 0.7 * 450.0)


def test_unreachable_strength_fraction_gives_nan():
    # 30% is below the 45% plateau, so it is never reached
    assert np.isnan(energy_for_strength_fraction(TRUE_PARAMS, 0.30))


def test_bootstrap_is_reproducible_and_has_five_columns():
    df = make_dataset()
    args = (df["impact_energy_J"].values, df["delam_area_mm2"].values,
            df["CAI_strength_MPa"].values)
    a = bootstrap_chain(*args, n_boot=30, seed=1)
    b = bootstrap_chain(*args, n_boot=30, seed=1)
    assert a.shape[1] == 5
    assert np.array_equal(a, b)


def test_confidence_band_is_ordered():
    params = np.array([TRUE_PARAMS, [5.5, 24.0, 440.0, 0.0042, 0.47],
                       [4.5, 26.0, 460.0, 0.0038, 0.43]])
    lo, hi = confidence_band(np.linspace(0, 40, 50), params)
    assert np.all(lo <= hi)
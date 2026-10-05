import numpy as np
from src.damage_models import (
    delamination_area, residual_strength, predict_strength_from_energy,
)


def test_no_damage_below_threshold():
    assert delamination_area(3.0, E_th=5.0, c=25.0) == 0.0


def test_damage_area_increases_with_energy():
    E = np.array([6.0, 10.0, 20.0, 40.0])
    A = delamination_area(E, E_th=5.0, c=25.0)
    assert np.all(np.diff(A) > 0)


def test_undamaged_strength_equals_sigma0():
    assert np.isclose(residual_strength(0.0, 450.0, 0.004, 0.45), 450.0)


def test_strength_decreases_with_damage():
    A = np.array([0.0, 100.0, 300.0, 800.0])
    s = residual_strength(A, 450.0, 0.004, 0.45)
    assert np.all(np.diff(s) < 0)


def test_strength_never_below_plateau():
    s = residual_strength(1e6, 450.0, 0.004, 0.45)
    assert s >= 450.0 * 0.45 - 1e-9


def test_full_chain_is_consistent():
    E = 20.0
    A = delamination_area(E, 5.0, 25.0)
    expected = residual_strength(A, 450.0, 0.004, 0.45)
    assert np.isclose(predict_strength_from_energy(E, 5.0, 25.0, 450.0, 0.004, 0.45), expected)
    
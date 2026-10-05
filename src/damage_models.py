"""Empirical models linking impact energy, delamination damage and
residual (compression-after-impact) strength of CFRP laminates.

These are simple empirical forms, chosen to capture the main physics:
  1. Below a threshold energy, little or no delamination occurs.
  2. Above it, damage area grows roughly in proportion to the excess energy.
  3. Residual strength falls as damage grows, but levels off at a plateau
     (the laminate does not lose all of its strength).
"""

import numpy as np


def delamination_area(E, E_th, c):
    """Projected delamination area (mm^2) after an impact of energy E (J).

    A = c * (E - E_th) for E > E_th, otherwise 0.
    E_th: threshold energy (J).  c: damage growth rate (mm^2 per J).
    """
    E = np.asarray(E, dtype=float)
    return c * np.clip(E - E_th, 0.0, None)


def residual_strength(A, sigma0, beta, r):
    """Compression-after-impact strength (MPa) for delamination area A (mm^2).

    sigma = sigma0 * ( r + (1 - r) * exp(-beta * A) )
    sigma0: strength of undamaged laminate (MPa).
    beta:   how quickly strength drops with damage area (1/mm^2).
    r:      fraction of strength retained at very large damage (plateau).
    """
    A = np.asarray(A, dtype=float)
    return sigma0 * (r + (1.0 - r) * np.exp(-beta * A))


def predict_strength_from_energy(E, E_th, c, sigma0, beta, r):
    """Full chain: impact energy -> delamination area -> residual strength."""
    return residual_strength(delamination_area(E, E_th, c), sigma0, beta, r)
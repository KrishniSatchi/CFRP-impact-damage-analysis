# CFRP Impact Damage Analysis

Python workflow linking impact energy, delamination damage and
compression-after-impact (CAI) strength of CFRP laminates, with bootstrap
uncertainty quantification and an allowable-impact-energy calculation.

## Objectives

- Model the chain: impact energy → delamination area → residual strength
- Recover model parameters with 95% confidence intervals
- Quantify uncertainty (bootstrap confidence bands, prediction intervals)
- Estimate the impact energy a laminate can tolerate for a given strength target

## Methods

Python | NumPy | SciPy | pandas | Matplotlib | pytest

## Key findings

- Fitted threshold energy: [value] J (95% CI [low, high])
- At [value]% of undamaged strength, the allowable impact energy is
  [best estimate] J (conservative: [value] J)
- The physics-based model outperformed a straight line: RMSE [value] vs [value] MPa

## Results

![Residual strength with uncertainty](figures/uncertainty_band.png)
![Allowable energy](figures/allowable_energy.png)

[2-3 sentences of YOUR interpretation]

## Limitations

- Data in `data/synthetic_cfrp.csv` is **simulated** from a known model to
  validate the pipeline. It is not experimental data.
- Empirical model: ignores fibre breakage, layup and thickness effects.
- Uncertainty uses a bootstrap on a small dataset, not formal design allowables
  (e.g. B-basis).
- Results should not be extrapolated beyond the tested energy range.

## How to run

    pip install -r requirements.txt
    python -m pytest
    python -m src.generate_data

Then run `notebooks/01_analysis.ipynb` and `notebooks/02_uncertainty.ipynb`.

## Repository structure

    data/        synthetic dataset and documentation
    notebooks/   analysis and uncertainty notebooks
    src/         models, data generator, uncertainty tools
    figures/     output figures
    tests/       unit tests

## References

[Add any papers or standards you cite, e.g. ASTM D7136 / D7137]
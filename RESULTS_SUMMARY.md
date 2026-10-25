# Quantum-Dynamics Benchmark Summary

This document summarizes the local Forte-like simulations, mitigation tests,
Trotter-order comparisons, cost estimates, and generated result artifacts.

## Executive summary

The combined method—zero-noise extrapolation (ZNE), even-parity symmetry
postselection, and Pauli-frame debiasing—helped the native Ising problems but
did not help the XXZ chain at the tested 600-shot budget:

| Problem | Raw baseline RMS | Combined RMS | Outcome |
|---|---:|---:|---|
| Stage 1 | 0.073646 | **0.031613** | Improved |
| Stage 2 | 0.103183 | 0.126943 | Worse |
| Stage 3, 3×3 | 0.076588 | **0.029366** | Improved |
| Stage 3, 4×4 | 0.067646 | **0.040630** | Improved |

The 6×6 Stage 3 problem was intentionally excluded from local dense exact
simulation. It has no local exact reference in this project and is too large
for the requested workflow.

For the dedicated Stage 3 3×3 order study, order 3 combined with mitigation
performed best at both shot counts tested, although the result varied
substantially between independent 600-shot runs.

## Problems and references

- **Stage 1:** 12-qubit transverse-field Ising chain, exact reference,
  12 time points from 0.25 to 3.0.
- **Stage 2:** 12-qubit XXZ Heisenberg chain, exact reference, 12 time points
  from 0.25 to 3.0.
- **Stage 3 3×3:** 9-qubit transverse-field Ising lattice, exact rehearsal
  reference, 6 time points from 0.25 to 1.5.
- **Stage 3 4×4:** 16-qubit transverse-field Ising lattice, exact rehearsal
  reference, 6 time points from 0.25 to 1.5.
- **Stage 3 6×6:** 36-qubit transverse-field Ising lattice; excluded from
  dense local exact simulation.

All noisy local results use the repository's `forte_like_noise_model()`.
This is a rehearsal model based on published Forte-like fidelities, not a
prediction of a particular hardware calibration.

## Combined mitigation benchmark

The combined benchmark used:

1. Physical circuit folding at noise factors **1, 3, and 5**.
2. Even-parity postselection, matching the conserved symmetry of the initial
   state and Hamiltonian.
3. Twenty-five randomized Pauli-frame variants per time point and fold.
4. Linear extrapolation of the three folded expectation values to zero noise.

At 600 total shots per time point, the three folds and 25 variants receive
8 shots per physical circuit. The mitigation benchmark therefore trades
substantial circuit count and shot efficiency for reduced systematic error.

### Results

| Problem | Baseline RMS | Mitigated RMS | Relative change |
|---|---:|---:|---:|
| Stage 1 | 0.073646 | 0.031613 | 57.1% lower |
| Stage 2 | 0.103183 | 0.126943 | 23.0% higher |
| Stage 3 3×3 | 0.076588 | 0.029366 | 61.6% lower |
| Stage 3 4×4 | 0.067646 | 0.040630 | 39.9% lower |

Stage 2 contains three two-qubit interactions per bond (`XX`, `YY`, and
`ZZ`). Folding those circuits greatly increases two-qubit exposure, while
the fixed 600-shot budget leaves only a few shots per variant. In this test,
the variance and postselection loss outweighed the bias reduction.

### Error bars

The final benchmark JSON files contain count-derived one-standard-deviation
errors for both the raw baseline and combined result. Combined errors are
propagated through the linear ZNE intercept weights:

\[
\sigma_0^2 = \sum_k w_k^2\sigma_k^2.
\]

The earlier Stage 1 plotting script used conservative aggregate estimates;
those were superseded for the combined-stage results by actual
postselected-count standard errors and propagated ZNE uncertainties.

## Trotter-order study: Stage 3 3×3

Order 1 is the plain product formula and is used as the baseline. Order 2 is
the symmetric Strang construction. Order 3 uses the notebook's Ruth
coefficients:

```text
((7/24,  2/3),
 (3/4,  -2/3),
 (-1/24, 1.0))
```

Each order was tested both without mitigation and with the complete combined
method. These are independent noisy simulations against the same exact
reference.

### Independent 600-shot runs

| Method | Run 1 RMS | Run 2 RMS | Mean RMS |
|---|---:|---:|---:|
| Order 2, raw | 0.072307 | 0.071963 | 0.072135 |
| Order 2 + mitigation | 0.048381 | 0.038619 | **0.043500** |
| Order 3, raw | 0.098426 | 0.103800 | 0.101113 |
| Order 3 + mitigation | 0.025440 | 0.053147 | **0.039294** |

The raw order-3 result is worse than raw order 2 because the higher-order
circuit has more gates and therefore more noise. Mitigation substantially
reduces that error, but its run-to-run variation is visible at 600 shots.

### Independent 1,500-shot run

| Method | Raw RMS | Mitigated RMS | Improvement |
|---|---:|---:|---:|
| Order 2 | 0.067181 | 0.055800 | 16.9% |
| Order 3 | 0.098643 | **0.044313** | 55.1% |

The 1,500-shot run reduces statistical uncertainty. Order 3 remains noisier
without mitigation, but provides the best mitigated RMS in this test.

### Order 2 versus order 1

Order 2 is not the same circuit as order 1:

- For the Ising stages, both use the same number of two-qubit gates per
  Trotter step, but order 2 uses additional field-rotation layers.
- For Stage 2's XXZ implementation, order 2 approximately doubles the
  two-qubit interaction layers per step.
- Order 3 uses three coefficient sublayers per step and therefore has still
  greater circuit exposure.

Consequently, increasing Trotter order improves ideal product-formula error
but can worsen the unmitigated hardware error.

## Forte cost estimates

The notebook describes the estimate as

```text
max(minimum, shots × (1Q gates × rate1 + 2Q gates × rate2))
```

The following approximate rates reproduce the notebook's rounded Stage 3
baseline table:

```text
rate1 = $0.00003 per 1Q gate-shot
rate2 = $0.00060 per 2Q gate-shot
minimum = $20 per circuit
```

These are planning estimates only. IonQ bills post-compilation gate counts,
and the actual organization rate card should be checked with
`estimate_cost()` before submission.

### 600-shot Stage 3 3×3 order-3 run

The raw order-3 series has six circuits. Its gate totals are 567 one-qubit
and 756 two-qubit gates across the series. Applying the formula circuit by
circuit gives:

**Raw order 3: approximately `$288.92`.**

The combined order-3 run has:

- 3 physical folds;
- 25 Pauli-frame variants;
- 6 time points;
- 450 physical circuits total;
- 8 shots per physical circuit.

Every circuit reaches the `$20` minimum, giving:

**Order 3 + ZNE + symmetry + debiasing: approximately `$9,000`.**

The combined estimate is high primarily because the fixed 600-shot budget is
spread over 450 circuits. It is not comparable to a six-circuit raw series
unless the shot and circuit budget are explicitly included in the comparison.

### Debiasing-disabled cost option

The simulators now accept `--remove-debiasing`. This disables the 25 randomized
Pauli-frame variants while retaining ZNE and symmetry postselection. For the
same 600-shot Stage 3 3×3 order-3 calculation, this gives 18 physical
circuits—three folds across six time points—with 200 shots per circuit:

| Configuration | Approximate cost |
|---|---:|
| Raw order 3 | `$288.92` |
| ZNE + symmetry, debiasing removed | **`$888.83`** |
| ZNE + symmetry + Pauli-frame debiasing | `$9,000.00` |

The option is available on `software_debias_stage1.py`,
`combined_zne_stages.py`, `order_mitigation_benchmark.py`, and
`order_stage3_3x3.py`. Existing checkpoints are treated as mode-specific so a
run cannot silently reuse results produced with the opposite debiasing mode.

### 600-shot debiasing-disabled order run

An independent Stage 3 3×3 run was completed with debiasing disabled:

| Order | Raw RMS | ZNE + symmetry, no debiasing RMS |
|---|---:|---:|
| Order 2 | 0.052363 | 0.061275 |
| Order 3 | 0.096595 | **0.034089** |

The corresponding comparison against the earlier 600-shot debiasing-enabled
run is plotted in
`results/order_stage3_3x3_debias_comparison_600.png` and stored in
`results/order_stage3_3x3_debias_comparison_600.json`.

## Generated artifacts

### Combined-stage results and plots

```text
results/combined_zne_stages/stage1.json
results/combined_zne_stages/stage1.png
results/combined_zne_stages/stage2.json
results/combined_zne_stages/stage2.png
results/combined_zne_stages/stage3_3x3.json
results/combined_zne_stages/stage3_3x3.png
results/combined_zne_stages/stage3_4x4.json
results/combined_zne_stages/stage3_4x4.png
```

### Order-study results

```text
results/order_stage3_3x3.json
results/order_stage3_3x3_run2.json
results/order_stage3_3x3_comparison.json
results/order_stage3_3x3_1500.json
results/order_stage3_3x3_1500_comparison.json
```

The order plots were generated later at the user's request:

```text
results/order_stage3_3x3_orders_600.png
results/order_stage3_3x3_orders_600.json
results/order_stage3_3x3_orders.png
results/order_stage3_3x3_orders_plot.json
```

The 600-shot plot uses the 600-shot order-2/order-3 data; the other order
plot uses the 1,500-shot data. The plots include exact, order-1 baseline,
raw order-2/order-3, and mitigated order-2/order-3 curves with stored error
bars.

## Reproducibility and limitations

- The benchmark scripts checkpoint after each stage or order so interrupted
  simulations do not discard completed work.
- The repository test suite passed with **8 tests** during the benchmark work.
- The Pauli-frame debiasing is a local software approximation using randomized
  mathematically equivalent circuit variants. It is not IonQ's built-in native
  debiasing, which was disabled by the challenge rules.
- The local noise model omits hardware effects such as drift, crosstalk, and
  coherent calibration errors.
- The combined method's extrapolation error bars include propagated sampling
  uncertainty, but they do not quantify model error, extrapolation bias, or
  correlations between variants.
- RMS values summarize deviations from the exact reference over the listed
  time grid; they are not hardware validation of the 6×6 Stage 3 target.

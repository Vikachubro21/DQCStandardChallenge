# Validation run

This folder contains the next-stage validation artifacts for full Stage 1,
Stage 2, and full 6×6 Stage 3. The 3×3 and 4×4 rehearsals are excluded.

For every stage and Trotter order 1, 2, and 3, the generated circuits use:

- physical ZNE folds `1, 3, 5`;
- even-parity symmetry postprocessing;
- no Pauli-frame debiasing;
- no IonQ job submission.

The notebook shot counts were used as follows:

- Stage 1: 500 shots per time point;
- Stage 2: 500 shots per time point;
- Stage 3: 300 and 1000 shots per time point.

## Files

- `build_validation.py` generates the circuits and offline planning estimates.
- `generated_validation_circuits.py` contains the generated circuit registry.
- `validation_costs.json` contains the local planning estimates.
- `request_ionq_cost_estimates.py` requests only IonQ's free `jobs/estimate`
  endpoint.
- `ionq_cost_estimates.json` contains IonQ's returned estimates.

The IonQ request script performs only GET-style cost-estimate requests through
`estimate_cost()`. It does not call `backend.run()`, submit a job, consume
shots, or charge the project.

## IonQ estimate results

These totals include all three ZNE folds for every time point. The shot count
shown is the total per time point, divided across the three fold circuits.

| Stage | Order | 500 shots | 300 shots | 1000 shots |
|---|---:|---:|---:|---:|
| Stage 1 | 1 | $1,436.53 | — | — |
| Stage 1 | 2 | $1,594.99 | — | — |
| Stage 1 | 3 | $3,924.53 | — | — |
| Stage 2 | 1 | $4,703.49 | — | — |
| Stage 2 | 2 | $9,359.99 | — | — |
| Stage 2 | 3 | $28,032.49 | — | — |
| Stage 3 | 1 | — | $1,106.10 | $3,587.77 |
| Stage 3 | 2 | — | $1,189.06 | $3,876.66 |
| Stage 3 | 3 | — | $3,234.22 | $10,759.98 |

IonQ estimates are more authoritative than the local planning estimates,
because they account for the provider's current rate card and estimate
service. They can still differ from a final bill if compilation or service
conditions change.

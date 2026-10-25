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

## Gate counts and local planning costs

The following table includes all ZNE folds `1,3,5`, with measurements excluded.
Stage 3 refers to the full 6×6, 36-qubit problem. Costs use the local planning
assumptions in `validation_costs.json`:
`$0.00003` per 1Q gate-shot, `$0.00060` per 2Q gate-shot, and a `$20` minimum
per circuit.

| Stage | Order | 1Q gates | 2Q gates | Total gates | Shots | Estimated cost |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 1 | 8,424 | 7,722 | 16,146 | 500 | $1,034.29 |
| Stage 1 | 2 | 16,848 | 7,722 | 24,570 | 500 | $1,067.10 |
| Stage 1 | 3 | 25,272 | 23,166 | 48,438 | 500 | $2,493.47 |
| Stage 2 | 1 | 62,424 | 23,166 | 85,590 | 500 | $2,672.14 |
| Stage 2 | 2 | 124,200 | 46,332 | 170,532 | 500 | $5,250.77 |
| Stage 2 | 3 | 371,304 | 138,996 | 510,300 | 500 | $15,693.10 |
| Stage 3 6×6 | 1 | 6,804 | 11,340 | 18,144 | 300 | $755.53 |
| Stage 3 6×6 | 1 | 6,804 | 11,340 | 18,144 | 1,000 | $2,341.36 |
| Stage 3 6×6 | 2 | 13,608 | 11,340 | 24,948 | 300 | $773.46 |
| Stage 3 6×6 | 2 | 13,608 | 11,340 | 24,948 | 1,000 | $2,408.97 |
| Stage 3 6×6 | 3 | 20,412 | 34,020 | 54,432 | 300 | $2,111.31 |
| Stage 3 6×6 | 3 | 20,412 | 34,020 | 54,432 | 1,000 | $7,001.11 |

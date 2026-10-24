# Standard track: simulating quantum dynamics you can't check

Participant starter kit for the standard track. Fork it, clone your fork, and
work in `notebooks/Standard_Track_Challenge.ipynb`.

**The challenge.** Simulate the time evolution of three spin systems on IonQ's
**Forte 1** trapped-ion quantum computer, and estimate one observable at each
time as accurately as you can, with honest error bars:

| Stage | System | Qubits | Hardware mapping | Exact answer |
|---|---|---|---|---|
| 1 | Transverse-field Ising chain | 12 | native (one `rxx` per bond) | computable |
| 2 | XXZ Heisenberg chain | 12 | non-native (three entangling gates per bond) | computable |
| 3 | 2D transverse-field Ising, 6 × 6 | 36 | native | **not provided** |

Stages 1 and 2 teach you what your methods really achieve on hardware. Stage 3
asks the question current research is arguing about: once a quantum simulation
is beyond classical checking, **how do you justify trusting its answer?**

## Setup

Use Python 3.10 or newer (tested with 3.12, see `.python-version`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

No local setup? Open your fork in GitHub Codespaces; the `.devcontainer`
installs everything.

### IonQ access

1. Accept the IonQ invitation email and sign in at
   [cloud.ionq.com](https://cloud.ionq.com) with the invited address.
2. At [cloud.ionq.com/settings/keys](https://cloud.ionq.com/settings/keys),
   create an API key **in your team's project** `std-<your team ID>` (not your
   personal workspace). Copy it right away; it is shown only once.
3. Put it in a `.env` file in the repository root as `IONQ_API_KEY=...`.
   `.env` is git-ignored. Never commit a key or paste it into a notebook.

Without a key, the notebook still runs: all simulations are local and the IonQ
cells are skipped.

## Rules in brief

- Machine: IonQ Forte 1 (`qpu.forte-1`) only. IonQ's cloud simulator (up to 29
  qubits, ideal or `forte-1` noise model) is free.
- Any gates IonQ accepts. Circuits must be your team's own.
- **No debiasing** (IonQ's built-in error mitigation). The helper
  `run_without_debiasing` switches it off. Your own mitigation is encouraged.
- Each team project has a hard dollar cap, released in two halves (start of
  event and midnight).
- Hand in one results file per stage (`write_results`, notebook Section 6) and
  present your work.

## Package

`dynamics_challenge` holds the fixed problem definitions and helpers:

```python
from dynamics_challenge import PROBLEMS, exact_series, trotter_circuit, series_circuits
from dynamics_challenge import observable_from_counts, series_from_counts, forte_like_noise_model
from dynamics_challenge import write_results, compare_with_exact
from dynamics_challenge.ionq import IONQ_QPU, estimate_cost, run_without_debiasing, record_job

problem = PROBLEMS["stage1"]
print(problem.hamiltonian_text(), problem.times)
print(exact_series(problem))                       # exact answers (stages 1, 2 and the stage-3 rehearsals)
circuit = trotter_circuit(problem, time=1.0, steps=4)   # baseline, rx/ry/rz/rxx only
```

Qubit `i` is Qiskit qubit `i`; in counts keys the rightmost character is qubit 0.

`tests/` contains smoke tests (`python -m pytest tests`), and
`.github/workflows/smoke-test.yml` runs them plus the whole notebook (without
an IonQ key) on every push.

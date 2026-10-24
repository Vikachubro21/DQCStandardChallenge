"""The three fixed problems of the standard track.

Every number that defines a problem lives here, so all teams (and the judges)
work on exactly the same Hamiltonian, initial state, observable and time grid.

Qubit ``i`` is Qiskit qubit ``i``. In a counts key such as ``'0110'`` the
rightmost character is qubit 0 (Qiskit's little-endian convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Problem:
    """One time-evolution task: evolve ``initial_bits`` under ``H`` and track one observable."""

    key: str
    title: str
    model: str                      # "tfim" or "xxz"
    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    couplings: dict = field(hash=False)
    initial_bits: str               # qubit 0 first (NOT a Qiskit counts key)
    observable: str                 # "magnetization" or "staggered_magnetization"
    times: tuple[float, ...]
    exact_reference: bool           # True if the exact answer is computable on a laptop

    def hamiltonian_text(self) -> str:
        c = self.couplings
        if self.model == "tfim":
            return f"H = {c['J']} * sum_<ij> X_i X_j + {c['h']} * sum_i Z_i"
        return f"H = {c['J']} * sum_<ij> (X_i X_j + Y_i Y_j + {c['delta']} Z_i Z_j)"


def chain_edges(n: int) -> tuple[tuple[int, int], ...]:
    """Open chain 0-1-2-...-(n-1)."""
    return tuple((i, i + 1) for i in range(n - 1))


def square_lattice_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    """Open rows x cols square lattice; site (r, c) is qubit r * cols + c."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return tuple(edges)


TIMES_1D = tuple(round(0.25 * k, 2) for k in range(1, 13))    # 0.25, 0.50, ..., 3.00
TIMES_2D = tuple(round(0.25 * k, 2) for k in range(1, 7))     # 0.25, 0.50, ..., 1.50


def _ising_2d(key: str, title: str, rows: int, cols: int, exact: bool) -> Problem:
    return Problem(
        key=key,
        title=title,
        model="tfim",
        num_qubits=rows * cols,
        edges=square_lattice_edges(rows, cols),
        couplings={"J": 1.0, "h": 1.0},
        initial_bits="0" * (rows * cols),
        observable="magnetization",
        times=TIMES_2D,
        exact_reference=exact,
    )


PROBLEMS: dict[str, Problem] = {
    # Stage 1: every term maps to one native gate (rxx for XX, rz for Z).
    "stage1": Problem(
        key="stage1",
        title="Stage 1: transverse-field Ising chain, 12 qubits (native, exactly solvable)",
        model="tfim",
        num_qubits=12,
        edges=chain_edges(12),
        couplings={"J": 1.0, "h": 1.0},
        initial_bits="0" * 12,
        observable="magnetization",
        times=TIMES_1D,
        exact_reference=True,
    ),
    # Stage 2: same size, but each bond term XX + YY + delta*ZZ needs three entangling gates.
    "stage2": Problem(
        key="stage2",
        title="Stage 2: XXZ Heisenberg chain, 12 qubits (non-native, exactly solvable)",
        model="xxz",
        num_qubits=12,
        edges=chain_edges(12),
        couplings={"J": 1.0, "delta": 0.5},
        initial_bits="01" * 6,       # Neel state |0101...>, qubit 0 first
        observable="staggered_magnetization",
        times=TIMES_1D,
        exact_reference=True,
    ),
    # Stage 3: the native Ising model on a 6 x 6 lattice. No exact reference is provided.
    "stage3": _ising_2d(
        "stage3", "Stage 3: 2D transverse-field Ising model, 6 x 6 = 36 qubits (native, no reference)", 6, 6, False
    ),
    # Smaller copies of stage 3 that you CAN check exactly: use them to test your methods.
    "stage3_3x3": _ising_2d("stage3_3x3", "Stage 3 rehearsal: 3 x 3 = 9 qubits", 3, 3, True),
    "stage3_4x4": _ising_2d("stage3_4x4", "Stage 3 rehearsal: 4 x 4 = 16 qubits", 4, 4, True),
}


def get_problem(key: str) -> Problem:
    try:
        return PROBLEMS[key]
    except KeyError:
        raise KeyError(f"Unknown problem {key!r}; choose one of {sorted(PROBLEMS)}.") from None

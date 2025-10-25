from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit
import numpy as np
import time

def run(input_data, solver_params, extra_arguments):
    """
    Minimal QAOA-style quantum step:
    - Builds a circuit from a toy QUBO (Q)
    - Runs it on Quantinuum backend (offline handle)
    - Returns the most likely bitstring and diagnostics
    """

    # -------------------------
    # 0. (Placeholder) QUBO + params
    #    In the real pipeline, Q comes from Step 3 (build_qubo_revised)
    #    and beta/gamma come from Step 4 (warm_start_params).
    #    Here we hardcode a small Q for demo sanity.
    # -------------------------
    Q = np.array([
        [4.9987,   10.0,    10.0002],
        [10.0,     5.0,     10.0   ],
        [10.0002,  10.0,    5.0009 ]
    ], dtype=float)

    beta = 0.06283185307179585
    gamma = 1.5393804002589988

    n = Q.shape[0]

    # -------------------------
    # 1. Build QAOA-like circuit
    # -------------------------
    circ = Circuit(n)

    # Initialize in uniform superposition
    for q in range(n):
        circ.H(q)

    # Cost layer (phase separator)
    for i in range(n):
        # single-qubit term ~ Q[i,i] * Z_i
        circ.Rz(2 * gamma * Q[i, i], i)

        # pairwise coupling terms ~ Q[i,j] * Z_i Z_j
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                circ.CX(i, j)
                circ.Rz(2 * gamma * Q[i, j], j)
                circ.CX(i, j)

        # Mixer layer X rotations
        circ.Rx(2 * beta, i)

    # Measure all qubits
    circ.measure_all()

    # -------------------------
    # 2. Compile and execute on Quantinuum backend
    # -------------------------
    backend = QuantinuumWrapper.get_target()

    # Track runtime for reporting
    t0 = time.time()

    # Compile for device
    # backend.default_compilation_pass().apply(circ)
    compiled_circ = backend.get_compiled_circuit(circ)

    # Run circuit
    handle = backend.process_circuit(compiled_circ, n_shots=2000)
    result = backend.get_result(handle)
    counts = result.get_counts()

    runtime_s = time.time() - t0

    # -------------------------
    # 3. Extract best state + metadata
    # -------------------------
    # best_state: most frequent measured bitstring
    best_state = max(counts, key=counts.get)

    n_qubits = compiled_circ.n_qubits
    depth = compiled_circ.depth()

    # Debug prints (these will show in container logs/stdout)
    print("Most frequent bitstring:", best_state)
    print("Top results:", list(counts.items())[:3])
    print("Runtime (s):", runtime_s)
    print("n_qubits:", n_qubits)
    print("circuit_depth:", depth)

    # -------------------------
    # 4. Return JSON-compatible output
    # -------------------------
    return {
        "result": {
            "best_state": best_state,
            "counts_preview": dict(list(counts.items())[:3]),
            "runtime_s": runtime_s,
            "n_qubits": n_qubits,
            "circuit_depth": depth
        }
    }

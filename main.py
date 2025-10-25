from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit
import numpy as np


def run_qaoa(Q, beta, gamma, shots=2000):
    n = Q.shape[0]
    circ = Circuit(n)
    circ.H(range(n))

    for i in range(n):
        circ.Rz(2 * gamma * Q[i, i], i)
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                circ.CX(i, j)
                circ.Rz(2 * gamma * Q[i, j], j)
                circ.CX(i, j)
        circ.Rx(2 * beta, i)

    circ.measure_all()

    backend = QuantinuumWrapper.get_target()
    backend.default_compilation_pass().apply(circ)
    compiled_circ = backend.get_compiled_circuit(circ)
    handle = backend.process_circuit(compiled_circ, n_shots=shots)
    counts = backend.get_result(handle).get_counts()
    best_state = max(counts, key=counts.get)
    return best_state, counts


def run(input_data, solver_params, extra_arguments):
    
    ##### THIW IS HOW YOU READ INPUT DATA FROM JSON #####
    
    #for asset in input_data:
    # print (asset, input_data[asset]['name']
    #data
    #######################################################


    ##############################################
    ## THIS IS HOW YOU WILL ACCESS QUANTINUUM MACHINES ##
    backend = QuantinuumWrapper.get_target()
    
    circ = Circuit(2).H(0).CX(0, 1).CZ(0, 1)
    circ.measure_all()
    backend.default_compilation_pass().apply(circ)
    compiled_circ = backend.get_compiled_circuit(circ)
    handle = backend.process_circuit(compiled_circ, n_shots=3000)

    ## THE RETURN MUST BE JSON COMPATIBLE ##
    return {"result": str(backend.get_result(handle).get_counts())}


# Example usage of QAOA
if __name__ == "__main__":
    # Q matrix and parameters
    Q = np.array([[4.7993, 10.0, 10.0],
                  [10.0, 5.0001, 10.0],
                  [10.0, 10.0, 4.8784]])
    beta0 = 0.2416609733530611
    gamma0 = 1.4499658401183664
    
    bitstring, counts = run_qaoa(Q, beta0, gamma0)
    print("Most frequent bitstring:", bitstring)
    print("Top 3 results:", list(counts.items())[:3])

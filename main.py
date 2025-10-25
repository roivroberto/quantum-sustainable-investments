from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit
import numpy as np

def run(input_data, solver_params, extra_arguments):
    
    ##### THIW IS HOW YOU READ INPUT DATA FROM JSON #####
    
    #for asset in input_data:
    # print (asset, input_data[asset]['name']
    #data
    #######################################################

    # Q matrix and parameters
    Q = np.array([[4.7993, 10.0, 10.0],
                  [10.0, 5.0001, 10.0],
                  [10.0, 10.0, 4.8784]])
    beta = 0.2416609733530611
    gamma = 1.4499658401183664

    ##############################################
    ## THIS IS HOW YOU WILL ACCESS QUANTINUUM MACHINES ##
    
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
    handle = backend.process_circuit(compiled_circ, n_shots=2000)
    counts = backend.get_result(handle).get_counts()
    best_state = max(counts, key=counts.get)

    print("Most frequent bitstring:", best_state)
    print("Top 3 results:", list(counts.items())[:3])

    ## THE RETURN MUST BE JSON COMPATIBLE ##
    return {"result": str(backend.get_result(handle).get_counts())}

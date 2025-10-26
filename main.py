from quantinuum_wrapper import QuantinuumWrapper
from pytket import Circuit

def run(input_data, solver_params, extra_arguments):
    
    ##### THIW IS HOW YOU READ INPUT DATA FROM JSON #####
    
    #for asset in input_data:
    # print (asset, input_data[asset]['name']
    #data
    #######################################################


    ##############################################
    ## THIS IS HOW YOU WILL ACCESS QUANTINUUM MACHINES ##
    
    """
    Quantum advantage QAOA with multiple optimization strategies.
    """
    n = Q.shape[0]
    
    # Check qubit limit
    if n > 25:
        print(f"Using first 25 assets for quantum simulation")
        n = 25
        Q = Q[:n, :n]
        mu = mu[:n]
        Sigma = Sigma[:n, :n]
        alpha = alpha[:n]
    
    print(f"\n=== QUANTUM ADVANTAGE QAOA ===")
    print(f"Problem size: {n} qubits")
    
    # ✅ QUANTUM ADVANTAGE: Multiple parameter strategies
    strategies = [
        {"beta": 0.3, "gamma": 0.8, "p": 1, "name": "Alpha-focused"},
        {"beta": 0.5, "gamma": 0.6, "p": 2, "name": "Balanced"},
        {"beta": 0.7, "gamma": 0.4, "p": 1, "name": "Exploration"},
        {"beta": 0.4, "gamma": 0.9, "p": 2, "name": "Deep optimization"},
        {"beta": 0.6, "gamma": 0.7, "p": 3, "name": "Multi-layer"}
    ]
    
    best_result = None
    best_score = float('-inf')
    
    for strategy in strategies:
        print(f"Trying {strategy['name']}: beta={strategy['beta']:.1f}, gamma={strategy['gamma']:.1f}, p={strategy['p']}")
        
        # Build circuit
        circ = Circuit(n)
        
        # Initial superposition
        for i in range(n):
            circ.H(i)
        
        # QAOA layers
        for layer in range(strategy['p']):
            # Cost Hamiltonian
            for i in range(n):
                circ.Rz(2 * strategy['gamma'] * Q[i, i], i)
                for j in range(i + 1, n):
                    if abs(Q[i, j]) > 1e-6:
                        circ.CX(i, j)
                        circ.Rz(2 * strategy['gamma'] * Q[i, j], j)
                        circ.CX(i, j)
            
            # Mixer Hamiltonian
            for i in range(n):
                circ.Rx(2 * strategy['beta'], i)
        
        circ.measure_all()
        
        # Execute
        # backend = AerBackend()
        # compiled = backend.get_compiled_circuit(circ)
        # handle = backend.process_circuit(compiled, n_shots=shots)
        # result = backend.get_result(handle)
        # counts = result.get_counts()

        backend = QuantinuumWrapper.get_target()
        backend.default_compilation_pass().apply(circ)
        compiled_circ = backend.get_compiled_circuit(circ)
        handle = backend.process_circuit(compiled_circ, n_shots=2000)
        result = backend.get_result(handle)
        counts = result.get_counts()
        
        best_state = None
        best_sharpe_alpha = float('-inf')
        
        for state, count in counts.items():
            if count > 0:  # Only consider states that appeared
                bitstring = ''.join(map(str, state))
                x = np.array(list(map(int, bitstring)))
                
                if x.sum() > 0:  # At least one asset selected
                    # Normalize weights
                    w = x / x.sum()
                    
                    # Calculate portfolio metrics
                    ret = float(mu @ w)
                    risk = float(w @ Sigma @ w)
                    alpha_exposure = float(alpha @ w)
                    
                    # Sharpe-Alpha score (our target metric)
                    sharpe_ratio = ret / (np.sqrt(risk) + 1e-8)
                    sharpe_alpha = sharpe_ratio + alpha_exposure
                    
                    if sharpe_alpha > best_sharpe_alpha:
                        best_sharpe_alpha = sharpe_alpha
                        best_state = state
        
        if best_state and best_sharpe_alpha > best_score:
            best_score = best_sharpe_alpha
            best_result = (best_state, counts, best_sharpe_alpha)
            print(f"  New best Sharpe-Alpha: {best_sharpe_alpha:.4f}")
    
    if best_result:
        return best_result[0], best_result[1]
    else:
        # Fallback
        return (0,) * n, {}

    ## THE RETURN MUST BE JSON COMPATIBLE ##
    # return {"result": str(backend.get_result(handle).get_counts())}

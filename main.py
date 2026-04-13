import numpy as np
import pandas as pd
import json
import cvxpy as cp
from pytket import Circuit
from quantinuum_wrapper import QuantinuumWrapper

def preprocess_data(data, alpha_scores, max_assets=25, selection_method="balanced"):
    """
    Intelligently preprocess data and select the best companies for QAOA.
    """
    all_data = []
    
    for ticker, content in data.items():
        try:
            # Extract price data
            history = pd.DataFrame(content["History"]).T
            prices = history["Close"].astype(float)
            returns = prices.pct_change().dropna()
            
            # Extract ESG data
            try:
                env_score = content["Sustainability"]["esgScores"]["environmentScore"]
            except (KeyError, TypeError):
                env_score = 0
                
            # Get alpha score
            alpha_score = alpha_scores.get(ticker, 0)
            
            # Calculate metrics
            mean_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            all_data.append({
                'ticker': ticker,
                'mean_return': mean_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'alpha_score': alpha_score,
                'env_score': env_score,
                'prices': prices,
                'returns': returns
            })
            
        except Exception as e:
            continue
    
    df = pd.DataFrame(all_data)
    df = df.dropna()
    
    # Select companies based on method
    if selection_method == "balanced":
        df['composite_score'] = (
            0.25 * df['sharpe_ratio'] +
            0.25 * df['alpha_score'] +
            0.25 * df['env_score'] +
            0.25 * (1 - df['volatility'])
        )
        selected_df = df.nlargest(max_assets, 'composite_score')
    else:
        selected_df = df.nlargest(max_assets, 'alpha_score')
    
    # Extract selected data
    selected_tickers = selected_df['ticker'].tolist()
    selected_prices = {row['ticker']: row['prices'] for _, row in selected_df.iterrows()}
    
    # Calculate portfolio metrics
    df_prices = pd.DataFrame(selected_prices)
    returns = df_prices.pct_change(fill_method=None).dropna()
    mu = returns.mean().values
    Sigma = returns.cov().values
    alpha_vector = np.array([alpha_scores.get(ticker, 0) for ticker in selected_tickers])
    
    return mu, Sigma, alpha_vector, selected_tickers

def build_quantum_qubo(mu, Sigma, alpha, q=0.6, gamma=0.4, lam=0.5, k=5):
    """
    Build QUBO specifically optimized for quantum advantage.
    """
    n = len(mu)
    
    # Normalize for better quantum processing
    mu_norm = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)
    Sigma_norm = Sigma / (np.max(np.abs(Sigma)) + 1e-8)
    alpha_norm = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
    
    Q = np.zeros((n, n))
    
    for i in range(n):
        Q[i, i] -= 2.0 * alpha_norm[i]
        Q[i, i] -= 0.5 * mu_norm[i]
        Q[i, i] += 0.3 * Sigma_norm[i, i]
        Q[i, i] += lam * (1 - 2*k/n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if Sigma_norm[i, j] < 0:
                Q[i, j] -= 0.5 * abs(Sigma_norm[i, j])
            else:
                Q[i, j] += 0.2 * Sigma_norm[i, j]
    
    return Q

def run(input_data, solver_params, extra_arguments):
    """
    Main entry point for the QCentroid platform.
    """
    # Load alpha scores (can be mock or real)
    alpha_scores = {}
    try:
        alpha_df = pd.read_csv('data/esg_alpha_scores_2023-12-29.csv')
        alpha_scores = dict(zip(alpha_df['ticker'], alpha_df['ESG_Alpha_Score']))
    except:
        pass

    # Preprocess
    mu, Sigma, alpha, tickers = preprocess_data(input_data, alpha_scores)
    
    # Build QUBO
    Q = build_quantum_qubo(mu, Sigma, alpha)
    n = Q.shape[0]
    
    # QAOA parameters
    beta, gamma, p = 0.5, 0.6, 1
    if solver_params:
        beta = solver_params.get("beta", beta)
        gamma = solver_params.get("gamma", gamma)
        p = solver_params.get("p", p)

    # Build Circuit
    circ = Circuit(n)
    for i in range(n):
        circ.H(i)
    
    for layer in range(p):
        for i in range(n):
            circ.Rz(2 * gamma * Q[i, i], i)
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-6:
                    circ.CX(i, j)
                    circ.Rz(2 * gamma * Q[i, j], j)
                    circ.CX(i, j)
        for i in range(n):
            circ.Rx(2 * beta, i)
    
    circ.measure_all()
    
    # Execute on Quantinuum
    backend = QuantinuumWrapper.get_target()
    backend.default_compilation_pass().apply(circ)
    compiled_circ = backend.get_compiled_circuit(circ)
    handle = backend.process_circuit(compiled_circ, n_shots=500)
    result = backend.get_result(handle)
    counts = result.get_counts()
    
    # Format result for QCentroid
    # Convert counts (dict of tuples) to dict of strings
    formatted_counts = {}
    for state, count in counts.items():
        bitstring = ''.join(map(str, state))
        formatted_counts[bitstring] = count

    return json.dumps(formatted_counts)

if __name__ == "__main__":
    # Test execution
    with open('data/input.json') as f:
        dic = json.load(f)
    print(run(dic['data'], None, None))

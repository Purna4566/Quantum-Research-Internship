import time
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit import transpile
from qiskit_aer import AerSimulator

# ==========================================
# 1. SCALABLE MARKET GENERATOR
# ==========================================

def generate_market(n_assets):
    """Generates a random market with n_assets."""
    np.random.seed(42)  
    mu = np.random.uniform(0.02, 0.15, n_assets)
    A = np.random.rand(n_assets, n_assets)
    sigma = np.dot(A, A.T) * 0.05
    return mu, sigma

def build_qubo(n_assets, mu, sigma, budget):
    """Builds the QUBO matrix for the portfolio optimization problem."""
    penalty = 3.5
    Q = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(i, n_assets):
            if i == j:
                Q[i, i] = -mu[i] + (0.5 * sigma[i, i]) + penalty*(1 - 2 * budget)
            else:
                Q[i, j] = (0.5 * sigma[i, j]) + penalty
                Q[j, i] = Q[i, j]
    return Q

def qubo_to_pauli(Q):
    """Converts a QUBO matrix to a SparsePauliOp."""
    n = len(Q)
    pauli_list = []
    for i in range(n):
        for j in range(n):
            if i == j:
                weight = -Q[i, i]/2
                if weight != 0:
                    op = ['I'] * n ; op[i] = 'Z'
                    pauli_list.append((''.join(op[::-1]), weight))
            elif i < j:
                weight = (Q[i, j]/4)*2
                if weight != 0:
                    op = ['I'] * n ; op[i] = 'Z' ; op[j] = 'Z'
                    pauli_list.append((''.join(op[::-1]), weight))
                lin_weight = -(Q[i][j] / 4) * 2
                if lin_weight != 0:
                    op_i = ['I'] * n ; op_i[i] = 'Z'
                    op_j = ['I'] * n ; op_j[j] = 'Z'
                    pauli_list.append((''.join(op_i[::-1]), lin_weight))
                    pauli_list.append((''.join(op_j[::-1]), lin_weight))
                    return SparsePauliOp.from_list(pauli_list).simplify()

def build_mixer(n):
    return SparsePauliOp.from_list([(''.join(['X' if j == i else 'I' for j in range(n)][::-1]), 1.0) for i in range(n)])

# ==========================================
# 2. EVALUATION & METRICS
# ==========================================

def calculate_financial_metrics(bitstring, mu, sigma):
    """Calculates Return and Variance for a given portfolio."""
    x = np.array([int(bit) for bit in bitstring])
    expected_return = np.dot(mu, x)
    variance = np.dot(x.T, np.dot(sigma, x))
    return expected_return, variance

def run_qaoa_scaling_test(n_assets, p_layers):
    """Executes QAOA and gathers Week 4 metrics."""
    print(f"\n--- Initiating QAOA | Assets: {n_assets} | Depth: p={p_layers} ---")
    budget = n_assets // 2
    mu, sigma = generate_market(n_assets)
    Q = build_qubo(n_assets, mu, sigma, budget)
    cost_h = qubo_to_pauli(Q)
    mixer_h = build_mixer(n_assets)

    circuit = transpile(
        QAOAAnsatz(cost_operator=cost_h, mixer_operator=mixer_h, reps=p_layers), 
        backend=AerSimulator(), 
        optimization_level=1
    )
    estimator = StatevectorEstimator()

    start_time = time.time()
    def objective(params):
        return float(estimator.run([(circuit, cost_h, [params])]).result()[0].data.evs[0])
    
    res = minimize(objective, np.random.rand(2 * p_layers) * np.pi, method='COBYLA', options={'maxiter': 50})
    runtime = time.time() - start_time

    bound_circuit = circuit.assign_parameters(res.x)
    bound_circuit.measure_all()
    counts = StatevectorSampler().run([(bound_circuit,)]).result()[0].data.meas.get_counts()

    total_shots = sum(counts.values())
    feasible_shots = 0
    best_portfolio = None
    max_hits = 0

    for state, hits in counts.items():
        corrected_state = state[::-1]
        assets_bought = corrected_state.count('1')
        
        if assets_bought == budget:
            feasible_shots += hits
            if hits > max_hits:
                max_hits = hits
                best_portfolio = corrected_state
                
    feasibility_rate = (feasible_shots / total_shots) * 100
    if best_portfolio is None:
        best_portfolio = max(counts, key=counts.get)[::-1]
        
    ret, var = calculate_financial_metrics(best_portfolio, mu, sigma)
    
    print(f"Runtime:            {runtime:.2f} seconds")
    print(f"Final Energy:       {res.fun:.4f}")
    print(f"Feasibility Rate:   {feasibility_rate:.2f}% (States obeying budget)")
    print(f"Best Portfolio:     {best_portfolio}")
    print(f"Expected Return:    {ret:.4f} ({(ret*100):.2f}%)")
    print(f"Portfolio Variance: {var:.4f}")

# ==========================================
# 3. MASTER EXECUTION
# ==========================================

if __name__ == "__main__":
    asset_scales = [10, 20, 30] 
    depths = [1, 2, 3]
    for n in asset_scales:
        for p in depths:
            try:
                run_qaoa_scaling_test(n_assets= n, p_layers= p)
            except MemoryError:
                print(f"!!! MEMORY ERROR on {n} Assets. Your laptop cannot hold 2^{n} quantum states in RAM! Skipping...")
                break
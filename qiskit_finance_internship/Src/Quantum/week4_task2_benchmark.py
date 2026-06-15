import time
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit import transpile
from qiskit_aer import AerSimulator

# ==========================================
# 1. MARKET DATA & FINANCIAL MATH
# ==========================================
def generate_market(n_assets=10):
    np.random.seed(42)
    mu = np.random.uniform(0.02, 0.15, n_assets)
    A = np.random.rand(n_assets, n_assets)
    sigma = np.dot(A, A.T) * 0.05
    return mu, sigma

def calc_financials(x, mu, sigma):
    """Calculates Return, Variance, and Sharpe Ratio."""
    ret = np.dot(mu, x)
    var = np.dot(x.T, np.dot(sigma, x))
    # Assuming Risk-Free Rate is 0% for the Sharpe Ratio calculation
    sharpe = ret / np.sqrt(var) if var > 0 else 0
    return ret, var, sharpe

# ==========================================
# 2. THE COMPETITORS
# ==========================================

def solve_cvxpy(mu, sigma, budget):
    """The Ground Truth: Exact classical convex optimization."""
    start_time = time.time()
    n = len(mu)
    x = cp.Variable(n, boolean=True)
    objective = cp.Maximize(mu.T @ x - 0.5 * cp.quad_form(x, sigma))
    constraints = [cp.sum(x) == budget]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCIP)

    runtime = time.time() - start_time
    optimal_x = np.round(x.value).astype(int)
    return optimal_x, runtime

def solve_simulated_annealing(mu, sigma, budget, iterations=1000):
    """Classical Heuristic: Explores the landscape by simulating cooling metal."""
    start_time = time.time()
    n = len(mu)
    current_x = np.zeros(n, dtype=int)
    current_x[:budget] = 1
    np.random.shuffle(current_x)

    def cost(x):
        return -(np.dot(mu, x) - 0.5 * np.dot(x.T, np.dot(sigma, x)))
    
    current_cost = cost(current_x)
    best_x = np.copy(current_x)
    best_cost = current_cost
    temp = 10.0 
    cooling_rate = 0.99
    for i in range(iterations):
        new_x = np.copy(current_x)
        idx_1 = np.where(new_x == 1)[0]
        idx_0 = np.where(new_x == 0)[0]
        swap_out = np.random.choice(idx_1)
        swap_in = np.random.choice(idx_0)
        new_x[swap_out] = 0
        new_x[swap_in] = 1
        
        new_cost = cost(new_x)
        
        if new_cost < current_cost or np.random.rand() < np.exp((current_cost - new_cost) / temp):
            current_x = np.copy(new_x)
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = np.copy(current_x)
                
        temp *= cooling_rate

    runtime = time.time() - start_time
    return best_x, runtime
def build_qubo(n, mu, sigma, budget, penalty=3.5):
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: Q[i][j] = -mu[i] + (0.5 * sigma[i][i]) + penalty * (1 - 2 * budget)
            else: Q[i][j] = (0.5 * sigma[i][j]) + penalty
    return Q

def qubo_to_pauli(Q):
    n = len(Q)
    pauli_list = []
    for i in range(n):
        for j in range(n):
            if i == j and -Q[i][i]/2 != 0:
                op = ['I']*n; op[i] = 'Z'
                pauli_list.append((''.join(op[::-1]), -Q[i][i]/2))
            elif i < j:
                if (Q[i][j]/4)*2 != 0:
                    op = ['I']*n; op[i] = 'Z'; op[j] = 'Z'
                    pauli_list.append((''.join(op[::-1]), (Q[i][j]/4)*2))
                if -(Q[i][j]/4)*2 != 0:
                    op_i = ['I']*n; op_i[i] = 'Z'; pauli_list.append((''.join(op_i[::-1]), -(Q[i][j]/4)*2))
                    op_j = ['I']*n; op_j[j] = 'Z'; pauli_list.append((''.join(op_j[::-1]), -(Q[i][j]/4)*2))
    return SparsePauliOp.from_list(pauli_list).simplify()

def solve_qaoa(mu, sigma, budget, p_layers=2):
    start_time = time.time()
    n = len(mu)
    Q = build_qubo(n, mu, sigma, budget)
    cost_h = qubo_to_pauli(Q)
    mixer_h = SparsePauliOp.from_list([(''.join(['X' if j==i else 'I' for j in range(n)][::-1]), 1.0) for i in range(n)])
    
    circuit = transpile(QAOAAnsatz(cost_operator=cost_h, mixer_operator=mixer_h, reps=p_layers), backend=AerSimulator(), optimization_level=1)
    estimator = StatevectorEstimator()
    
    def objective(params):
        return float(estimator.run([(circuit, cost_h, [params])]).result()[0].data.evs[0])
    
    res = minimize(objective, np.random.rand(2 * p_layers) * np.pi, method='COBYLA', options={'maxiter': 50})
    
    bound_circuit = circuit.assign_parameters(res.x)
    bound_circuit.measure_all()
    counts = StatevectorSampler().run([(bound_circuit,)]).result()[0].data.meas.get_counts()
    
    # Filter for valid portfolios
    max_hits = 0
    best_portfolio = None
    for state, hits in counts.items():
        corr_state = state[::-1]
        if corr_state.count('1') == budget and hits > max_hits:
            max_hits = hits
            best_portfolio = corr_state
            
    if not best_portfolio:
        best_portfolio = max(counts, key=counts.get)[::-1]
        
    runtime = time.time() - start_time
    return np.array([int(b) for b in best_portfolio]), runtime

# ==========================================
# 3. MASTER BENCHMARK EXECUTION
# ==========================================
if __name__ == "__main__":
    n_assets = 10
    budget = 5
    mu, sigma = generate_market(n_assets)
    
    print(f"=== QUANTUM VS CLASSICAL BENCHMARK ({n_assets} ASSETS) ===\n")
    
    # 1. CVXPY (Ground Truth)
    x_cvxpy, t_cvxpy = solve_cvxpy(mu, sigma, budget)
    ret_c, var_c, sharpe_c = calc_financials(x_cvxpy, mu, sigma)
    
    # 2. Simulated Annealing
    x_sa, t_sa = solve_simulated_annealing(mu, sigma, budget)
    ret_sa, var_sa, sharpe_sa = calc_financials(x_sa, mu, sigma)
    
    # 3. QAOA (p=2)
    x_qaoa, t_qaoa = solve_qaoa(mu, sigma, budget, p_layers=2)
    ret_q, var_q, sharpe_q = calc_financials(x_qaoa, mu, sigma)
    
    # Calculate Approximation Ratios
    approx_sa = (sharpe_sa / sharpe_c) * 100
    approx_qaoa = (sharpe_q / sharpe_c) * 100
    
    print(f"{'Algorithm':<20} | {'Runtime':<10} | {'Sharpe Ratio':<15} | {'Return':<10} | {'Approx Ratio':<12}")
    print("-" * 75)
    print(f"{'CVXPY (Exact)':<20} | {t_cvxpy:<8.4f} s | {sharpe_c:<15.4f} | {ret_c*100:>5.2f}%    | {'100.00%':<12}")
    print(f"{'Simulated Annealing':<20} | {t_sa:<8.4f} s | {sharpe_sa:<15.4f} | {ret_sa*100:>5.2f}%    | {approx_sa:>6.2f}%")
    print(f"{'QAOA (p=2)':<20} | {t_qaoa:<8.4f} s | {sharpe_q:<15.4f} | {ret_q*100:>5.2f}%    | {approx_qaoa:>6.2f}%")
    
    print("\nSelected Portfolios:")
    print(f"CVXPY: {x_cvxpy}")
    print(f"SA:    {x_sa}")
    print(f"QAOA:  {x_qaoa}")
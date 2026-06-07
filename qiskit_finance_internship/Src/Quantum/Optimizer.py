import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as StatevectorEstimator
from qiskit.primitives import StatevectorSampler
from qiskit import transpile
from qiskit_aer import AerSimulator

# IMPORTANT: If this import fails, run `pip install qiskit-algorithms` in your terminal
from qiskit_algorithms.optimizers import SPSA 

# ==========================================
# MODULE 1: PORTFOLIO QUBO LOADER
# ==========================================
def generate_10_asset_market():
    """Generates a QUBO for a 10-asset portfolio optimization problem."""
    np.random.seed(42)  
    n_assets = 10
    mu = np.random.uniform(0.02, 0.15, n_assets)  
    A = np.random.rand(n_assets, n_assets)  
    sigma = np.dot(A, A.T) * 0.05
    return n_assets, mu, sigma

def build_qubo_matrix(n_assets, mu, sigma, risk_aversion=0.5, budget=5, penalty=3.5):
    """Builds the QUBO matrix for the portfolio optimization problem."""
    Q = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                Q[i][j] = -mu[i] + (risk_aversion * sigma[i][i]) + penalty * (1 - 2 * budget)
            else:
                Q[i][j] = (risk_aversion * sigma[i][j]) + penalty
    return Q

n_assets, mu, sigma = generate_10_asset_market()
Q_matrix = build_qubo_matrix(n_assets, mu, sigma)
print("Module 1 Complete: 10-Asset QUBO Matrix Generated.")

# ==========================================
# MODULE 2: PROBLEM HAMILTONIAN BUILDER   
# ==========================================
def qubo_to_pauli_hamiltonian(Q):
    """Converts a classical QUBO matrix into a Qiskit Pauli-Z Hamiltonian."""
    n = len(Q)
    pauli_list = []
    
    for i in range(n):
        for j in range(n):
            if i == j:
                weight = -Q[i][i] / 2
                if weight != 0:
                    op_string = ['I'] * n
                    op_string[i] = 'Z'
                    pauli_list.append((''.join(op_string[::-1]), weight))
            elif i < j:
                # FIXED: ZZ weight is positive, individual Z weights are negative
                weight = (Q[i][j] / 4) * 2
                if weight != 0:
                    op_string = ['I'] * n
                    op_string[i] = 'Z'
                    op_string[j] = 'Z'
                    pauli_list.append((''.join(op_string[::-1]), weight))

                lin_weight = -(Q[i][j] / 4) * 2
                if lin_weight != 0:
                    op_str_i = ['I'] * n
                    op_str_i[i] = 'Z'
                    pauli_list.append((''.join(op_str_i[::-1]), lin_weight))
                    
                    op_str_j = ['I'] * n
                    op_str_j[j] = 'Z'
                    pauli_list.append((''.join(op_str_j[::-1]), lin_weight))

    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian.simplify()

cost_hamiltonian = qubo_to_pauli_hamiltonian(Q_matrix)
print("Module 2 Complete: QUBO Matrix Converted to Quantum Hamiltonian.")

# ==========================================
# MODULE 3: MIXER HAMILTONIAN BUILDER
# ==========================================
def build_mixer_hamiltonian(n):
    """Builds the standard X-mixer Hamiltonian for QAOA."""
    pauli_list = []
    for i in range(n):
        op_string = ['I'] * n
        op_string[i] = 'X'
        pauli_list.append((''.join(op_string[::-1]), 1.0))
    return SparsePauliOp.from_list(pauli_list)

mixer_hamiltonian = build_mixer_hamiltonian(n_assets)
print("Module 3 Complete: Mixer Hamiltonian Built.")

# ==========================================
# MODULE 4: QAOA CIRCUIT GENERATOR       
# ==========================================
def build_qaoa_circuit(cost_hamiltonian, mixer_hamiltonian, p):
    """Generates a parameterized QAOA circuit."""
    qaoa_circuit = QAOAAnsatz(
        cost_operator=cost_hamiltonian,
        mixer_operator=mixer_hamiltonian,
        reps=p
    )
    backend = AerSimulator()
    compiled_circuit = transpile(qaoa_circuit, backend=backend, optimization_level=1)
    return compiled_circuit

# ==========================================
# MODULE 5 & TASK 2: DEPTH EXECUTION LOOP
# ==========================================
def optimize_qaoa(circuit, cost_hamiltonian, p):
    """Optimizes the QAOA parameters using COBYLA."""
    estimator = StatevectorEstimator()
    
    def objective(params):
        pub = (circuit, cost_hamiltonian, [params])
        job = estimator.run([pub])
        return float(job.result()[0].data.evs[0]) 
    
    initial_params = np.random.rand(2 * p) * np.pi
    result = minimize(objective, initial_params, method='COBYLA', options={'maxiter': 100, 'disp': False})
    return result

def extract_portfolio(circuit, optimal_params):
    """Samples the circuit to find the winning portfolio."""
    bound_circuit = circuit.assign_parameters(optimal_params)
    bound_circuit.measure_all()

    sampler = StatevectorSampler()
    job = sampler.run([(bound_circuit,)])
    counts = job.result()[0].data.meas.get_counts()
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  Top 3 Quantum Measurements:")
    for state, count in sorted_counts[:3]:
        corrected_state = state[::-1]
        assets_bought = corrected_state.count('1')
        print(f"  [{corrected_state}] | Assets: {assets_bought} | Hits: {count}")
    
    return sorted_counts[0][0][::-1]

print("\n=== [TASK 2] QAOA DEPTH SCALING TEST ===")
for p in [1, 2, 3]:
    print(f"\n--- Testing Circuit Depth p={p} ---")
    circuit = build_qaoa_circuit(cost_hamiltonian, mixer_hamiltonian, p)
    result = optimize_qaoa(circuit, cost_hamiltonian, p)
    
    best_portfolio = extract_portfolio(circuit, result.x)
    print(f"  Final Energy: {result.fun:.4f}")

# ==========================================
# TASK 3: OPTIMIZER BENCHMARK
# ==========================================
def benchmark_optimizers(p_layer=2):
    """Runs a side-by-side benchmark of COBYLA, Nelder-Mead, and SPSA."""
    print("\n=== [TASK 3] OPTIMIZER BENCHMARK (p=2) ===")
    
    circuit = build_qaoa_circuit(cost_hamiltonian, mixer_hamiltonian, p=p_layer)
    estimator = StatevectorEstimator()
    initial_params = np.random.rand(2 * p_layer) * np.pi
    
    optimizers = ['COBYLA', 'Nelder-Mead', 'SPSA']
    results_table = {}

    for opt in optimizers:
        print(f"Running {opt}...")
        energy_history = []
        
        def objective(params):
            pub = (circuit, cost_hamiltonian, [params])
            job = estimator.run([pub])
            energy = float(job.result()[0].data.evs[0])
            energy_history.append(energy)
            return energy

        start_time = time.time()
        
        if opt in ['COBYLA', 'Nelder-Mead']:
            res = minimize(objective, initial_params, method=opt, options={'maxiter': 100})
            final_energy = res.fun
        elif opt == 'SPSA':
            spsa = SPSA(maxiter=100)
            res = spsa.minimize(objective, initial_params)
            final_energy = res.fun

        runtime = time.time() - start_time
        
        results_table[opt] = {
            "Final Energy": round(final_energy, 4),
            "Runtime (s)": round(runtime, 2),
            "History": energy_history
        }
        
        print(f" -> Done! Time: {runtime:.2f}s | Energy: {final_energy:.4f}")

    return results_table


benchmark_data = benchmark_optimizers(p_layer=2)
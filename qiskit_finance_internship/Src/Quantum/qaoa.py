import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as StatevectorEstimator
from scipy.optimize import minimize
from qiskit.primitives import StatevectorSampler
from qiskit import transpile
from qiskit_aer import AerSimulator

# ==========================================
# MODULE 1: PORTFOLIO QUBO LOADER
# ==========================================

def generate_10_asset_market():
    """Generates a QUBO for a 10-asset portfolio optimization problem."""
    # For simplicity, we create a random QUBO matrix
    np.random.seed(42)  # For reproducibility
    n_assets = 10
    mu = np.random.uniform(0.02, 0.15, n_assets)  # Expected returns
    A = np.random.rand(n_assets, n_assets)  
    sigma = np.dot(A, A.T)*0.05
    
    return n_assets, mu, sigma

def build_qubo_matrix(n_assets, mu, sigma, risk_aversion=0.5, budget=5, penalty=3.5):
    """Builds the QUBO matrix for the portfolio optimization problem."""
    Q = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                # Linear terms: Returns + Variance + Constraint Penalty
                Q[i][j] = -mu[i] + (risk_aversion * sigma[i][i]) + penalty * (1 - 2 * budget)
            else:
                # Quadratic terms: Covariance + Constraint Penalty
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
                weight = -Q[i][j]/2
                if weight != 0:
                    op_string = ['I'] * n
                    op_string[i] = 'Z'
                    pauli_list.append((''.join(op_string[::-1]), weight))
            elif i < j:
                weight = -(Q[i][j]/4)*2
                if weight != 0:
                    op_string = ['I'] * n
                    op_string[i] = 'Z'
                    op_string[j] = 'Z'
                    pauli_list.append((''.join(op_string[::-1]), weight))

                    lin_weight = Q[i][j]/4
                    if lin_weight != 0:
                        op_str_i = ['I'] * n
                        op_str_i[i] = 'Z'
                        pauli_list.append((''.join(op_str_i[::-1]), lin_weight))
                        
                        op_str_j = ['I'] * n
                        op_str_j[j] = 'Z'
                        pauli_list.append((''.join(op_str_j[::-1]), lin_weight))


# Combine all terms into a single quantum operator
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian.simplify()

cost_hamiltonian = qubo_to_pauli_hamiltonian(Q_matrix)
print("Module 2 Complete: QUBO Matrix Converted to Quantum Hamiltonian.")
print(f"Number of Pauli Strings: {len(cost_hamiltonian)}")

# ==========================================
# MODULE 3: Mixer Hamiltonian Builder
# ==========================================

def build_mixer_hamiltonian(n):
    """Builds the standard X-mixer Hamiltonian for QAOA."""
    pauli_list = []
    for i in range(n):
        op_string = ['I'] * n
        op_string[i] = 'X'
        pauli_list.append((''.join(op_string[::-1]), 1.0))
    mixer_hamiltonian = SparsePauliOp.from_list(pauli_list)
    return mixer_hamiltonian

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
circuit_p1 = build_qaoa_circuit(cost_hamiltonian, mixer_hamiltonian, p=1)
print("Module 4 Complete: QAOA Circuit Generated.")
print(f"Number of Parameters to Optimize: {circuit_p1.num_parameters}")

# ==========================================
# MODULE 5: Optimization Loop
# ==========================================
def optimize_qaoa(circuit, cost_hamiltonian, p):
    """Optimizes the QAOA parameters using a classical optimizer."""
    estimator = StatevectorEstimator()
    
    iteration = [0] 

    iteration = [0] 

    def objective(params):
        pub = (circuit, cost_hamiltonian, [params])
        job = estimator.run([pub])
        energy = job.result()[0].data.evs
        
        # Print EVERY step so we can see if it freezes
        iteration[0] += 1
        print(f"    -> COBYLA Guess {iteration[0]} | Energy: {float(energy[0]):.4f}")
            
        return float(energy[0]) 
    
    initial_params = np.random.rand(2 * p) * np.pi
    
    # ADDED: options={'maxiter': 100} to force the loop to stop after 100 tries
    result = minimize(
        objective, 
        initial_params, 
        method='COBYLA', 
        options={'maxiter': 100, 'disp': False} 
    )
    return result

# ==========================================
# FINAL EXECUTION & PORTFOLIO DECODING
# ==========================================
def extract_portfolio(circuit, optimal_params):
    """Binds optimal parameters and samples the circuit to find the winning portfolio."""
    bound_circuit = circuit.assign_parameters(optimal_params)
    bound_circuit.measure_all()

    sampler = StatevectorSampler()
    pub = (bound_circuit,)
    job = sampler.run([pub])
    counts = job.result()[0].data.meas.get_counts()
    best_state_binary = max(counts, key=counts.get)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Top 5 Quantum Measurements ---")
    for state, count in sorted_counts[:5]:
        # Reverse string to match our Little-Endian mapping
        corrected_state = state[::-1]
        
        # Count how many assets were actually selected to see if it obeyed the budget
        assets_bought = corrected_state.count('1')
        print(f"Portfolio: {corrected_state} | Assets Bought: {assets_bought} | Occurrences: {count}")
    
    # Return the absolute highest
    best_state_binary = sorted_counts[0][0]
    return best_state_binary[::-1]
print("\n=== INITIATING QAOA PORTFOLIO OPTIMIZATION ===")

for p in [1, 2, 3]:
    print(f"\n--- Testing Circuit Depth p={p} ---")
    
    circuit = build_qaoa_circuit(cost_hamiltonian, mixer_hamiltonian, p)
    result = optimize_qaoa(circuit, cost_hamiltonian, p)
    
    optimal_energy = result.fun
    optimal_angles = result.x
    best_portfolio = extract_portfolio(circuit, optimal_angles)
    
    print(f"Final Cost (Energy): {optimal_energy:.4f}")
    print(f"Optimal Parameters:  {optimal_angles}")
    print(f"Selected Portfolio:  {best_portfolio}")

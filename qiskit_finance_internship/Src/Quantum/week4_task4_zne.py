import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

# ==========================================
# 1. BASELINE MARKET & HAMILTONIAN SETUP
# ==========================================
def generate_market(n_assets=6):
    np.random.seed(42)
    mu = np.random.uniform(0.02, 0.15, n_assets)
    A = np.random.rand(n_assets, n_assets)
    sigma = np.dot(A, A.T) * 0.05
    return mu, sigma

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

# ==========================================
# 2. HARDWARE NOISE & UNITARY FOLDING
# ==========================================
def generate_medium_noise():
    """Generates standard commercial NISQ hardware noise."""
    noise_model = NoiseModel()
    err_1q = depolarizing_error(0.01, 1) # 1% single-qubit error
    err_2q = depolarizing_error(0.05, 2) # 5% two-qubit error
    noise_model.add_all_qubit_quantum_error(err_1q, ['u1', 'u2', 'u3', 'x', 'h', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(err_2q, ['cx', 'ecr', 'cz'])
    return noise_model
def fold_circuit(bound_circuit, scale_factor):
    """
    Artificially scales noise by folding the circuit.
    Scale 1: C
    Scale 3: C * C_dagger * C
    Scale 5: C * C_dagger * C * C_dagger * C
    """
    if scale_factor == 1:
        return bound_circuit
        
    folded = bound_circuit.copy()
    inverse_circ = bound_circuit.inverse()
    
    num_folds = (scale_factor - 1) // 2
    for _ in range(num_folds):
        folded.compose(inverse_circ, inplace=True)
        folded.compose(bound_circuit, inplace=True)
        
    return folded
# ==========================================
# 3. EXPERIMENT PIPELINE
# ==========================================
if __name__ == "__main__":
    n_assets = 6
    budget = 3
    p_layers = 1
    
    print("=== ZERO NOISE EXTRAPOLATION (ZNE) ===")
    
    # 1. Setup Math
    mu, sigma = generate_market(n_assets)
    Q = build_qubo(n_assets, mu, sigma, budget)
    cost_h = qubo_to_pauli(Q)
    mixer_h = SparsePauliOp.from_list([(''.join(['X' if j==i else 'I' for j in range(n_assets)][::-1]), 1.0) for i in range(n_assets)])
    
    ansatz = QAOAAnsatz(cost_operator=cost_h, mixer_operator=mixer_h, reps=p_layers)
    
    # 2. Find Ideal Target (The Ground Truth Vacuum)
    print("\n1. Calculating Ideal Target (0x Noise)...")
    ideal_estimator = AerEstimator()
    ideal_circuit = transpile(ansatz, backend=AerSimulator(), optimization_level=1)
    
    def objective(params):
        return float(ideal_estimator.run([(ideal_circuit, cost_h, [params])]).result()[0].data.evs[0])
    
    ideal_res = minimize(objective, np.random.rand(2 * p_layers) * np.pi, method='COBYLA', options={'maxiter': 30})
    optimal_angles = ideal_res.x
    true_ideal_energy = ideal_res.fun
    bound_ideal_circuit = ideal_circuit.assign_parameters(optimal_angles)
    
    # 3. Setup Noisy Hardware
    print("2. Firing up Noisy Hardware Backend...")
    noise_model = generate_medium_noise()
    noisy_backend = AerSimulator(noise_model=noise_model)
    noisy_estimator = BackendEstimatorV2(backend=noisy_backend)
    
    # 4. Execute Unitary Folding at scales 1, 3, and 5
    noise_scales = [1, 3, 5]
    measured_energies = []
    
    print("\n3. Running Scaled Noise Executions...")
    print(f"{'Scale Factor':<15} | {'Measured Energy':<15} | {'Signal Decay (Delta from Ideal)'}")
    print("-" * 75)
    
    for scale in noise_scales:
        # Fold the math, then compile it to physical gates
        folded_circ = fold_circuit(bound_ideal_circuit, scale)
        compiled_folded = transpile(folded_circ, backend=noisy_backend, optimization_level=0)
        
        # We don't pass parameters here because the circuit is already bound!
        pub = (compiled_folded, cost_h)
        
        # Safely extract the energy whether Qiskit returns a scalar or an array
        raw_evs = noisy_estimator.run([pub]).result()[0].data.evs
        energy = float(np.asarray(raw_evs).flatten()[0])
        
        measured_energies.append(energy)
        
        decay = abs(energy - true_ideal_energy)
        print(f"{scale}x             | {energy:<15.4f} | +{decay:.4f}")
        
    # 5. Richardson Extrapolation (Quadratic Fit back to 0)
    # Fit a parabola (degree 2) to the data points: x=scales, y=energies
    poly_coeffs = np.polyfit(noise_scales, measured_energies, 2)
    
    # Extrapolate to x = 0
    extrapolated_energy = np.polyval(poly_coeffs, 0)
    
    print("\n" + "=" * 50)
    print("=== EXTRAPOLATION RESULTS ===")
    print(f"True Ideal Energy (Vacuum):      {true_ideal_energy:.4f}")
    print(f"Raw Noisy Energy  (1x Noise):    {measured_energies[0]:.4f}")
    print(f"ZNE Mitigated Energy (0x Noise): {extrapolated_energy:.4f}")
    
    # Calculate recovery percentage
    total_error = abs(measured_energies[0] - true_ideal_energy)
    mitigated_error = abs(extrapolated_energy - true_ideal_energy)
    
    if total_error > 0:
        recovery = max(0, 100 * (1 - (mitigated_error / total_error)))
        print(f"\nError Recovered by ZNE: {recovery:.2f}%")
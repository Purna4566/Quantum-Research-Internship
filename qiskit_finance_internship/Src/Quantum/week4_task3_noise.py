import time
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2

# Import Qiskit Aer's noise module components
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# ==========================================
# 1. BASELINE MARKET & HAMILTONIAN SETUP
# ==========================================
def generate_market(n_assets=10):
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
# 2. NISQ HARDWARE NOISE MODEL BUILDER
# ==========================================
def generate_noise_scenario(scenario_name):
    """Builds a custom Qiskit Aer NoiseModel based on target NISQ constraints."""
    if scenario_name == "Scenario A (Ideal)":
        return None
        
    noise_model = NoiseModel()
    
    if scenario_name == "Scenario B (Low Noise)":
        gate_1q_err = 0.001   # 0.1% error
        gate_2q_err = 0.01    # 1.0% error
        readout_err = 0.01    # 1.0% error
    elif scenario_name == "Scenario C (Medium Noise)":
        gate_1q_err = 0.01    # 1.0% error
        gate_2q_err = 0.05    # 5.0% error
        readout_err = 0.05    # 5.0% error
    elif scenario_name == "Scenario D (High Noise)":
        gate_1q_err = 0.05    # 5.0% error
        gate_2q_err = 0.15    # 15.0% error
        readout_err = 0.15    # 15.0% error
    else:
        return None
    # 1. Define Depolarizing Error Channels
    err_1q = depolarizing_error(gate_1q_err, 1)
    err_2q = depolarizing_error(gate_2q_err, 2)
    
    # Apply to standard basis gates
    noise_model.add_all_qubit_quantum_error(err_1q, ['u1', 'u2', 'u3', 'x', 'h', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(err_2q, ['cx', 'ecr', 'cz'])
    
    # 2. Define Thermal Readout Error Matrices
    # Maps the probability of measuring 0 instead of 1, and vice versa
    error_ro = ReadoutError([[1.0 - readout_err, readout_err], [readout_err, 1.0 - readout_err]])
    noise_model.add_all_qubit_readout_error(error_ro)
    
    return noise_model

# ==========================================
# 3. EXPERIMENT PIPELINE
# ==========================================
if __name__ == "__main__":
    n_assets = 10
    budget = 5
    p_layers = 2
    
    mu, sigma = generate_market(n_assets)
    Q = build_qubo(n_assets, mu, sigma, budget)
    cost_h = qubo_to_pauli(Q)
    mixer_h = SparsePauliOp.from_list([(''.join(['X' if j==i else 'I' for j in range(n_assets)][::-1]), 1.0) for i in range(n_assets)])

    ideal_backend = AerSimulator()
    ideal_circuit = transpile(QAOAAnsatz(cost_operator=cost_h, mixer_operator=mixer_h, reps=p_layers), backend=ideal_backend, optimization_level=1)
    ideal_estimator = AerEstimator()
    def objective(params):
        return float(ideal_estimator.run([(ideal_circuit, cost_h, [params])]).result()[0].data.evs[0])
    
    ideal_res = minimize(objective, np.random.rand(2 * p_layers) * np.pi, method='COBYLA', options={'maxiter': 40})
    optimal_angles = ideal_res.x
    print("Calibration Complete. Evaluating Noise Scenarios...\n")

    scenarios = [
        "Scenario A (Ideal)",
        "Scenario B (Low Noise)",
        "Scenario C (Medium Noise)",
        "Scenario D (High Noise)"
    ]
    
    print(f"{'Hardware Environment':<26} | {'Expectation (Energy)':<22} | {'Feasibility Rate':<18}")
    print("-" * 75)
    for sc in scenarios:
        noise_model = generate_noise_scenario(sc)
        
        # Instantiate backend wrapper with targeted noise settings
        if noise_model:
            backend = AerSimulator(noise_model=noise_model)
        else:
            backend = AerSimulator()
            
        # 1. Build & Compile circuit 
        ansatz = QAOAAnsatz(cost_operator=cost_h, mixer_operator=mixer_h, reps=p_layers)
        compiled_circuit = transpile(ansatz, backend=backend, optimization_level=1)
        
        # 2. Run Noisy Expectation Estimation (PLUGGED INTO BACKEND)
        estimator = BackendEstimatorV2(backend=backend)
        try:
            pub_est = (compiled_circuit, cost_h, [optimal_angles])
            noisy_energy = float(estimator.run([pub_est]).result()[0].data.evs[0])
        except Exception:
            noisy_energy = np.nan
            
        # 3. Run Noisy Sampling Execution (PLUGGED INTO BACKEND)
        compiled_circuit.measure_all()
        
        sampler = BackendSamplerV2(backend=backend)
        pub_samp = (compiled_circuit, [optimal_angles])
        
        result_data = sampler.run([pub_samp]).result()[0].data
        
        if hasattr(result_data, 'meas'):
            counts = result_data.meas.get_counts()
        else:
            counts = list(result_data.values())[0].get_counts()
        
        total_shots = sum(counts.values())
        feasible_shots = 0
        
        for state, hits in counts.items():
            corrected_state = state[::-1]
            if corrected_state.count('1') == budget:
                feasible_shots += hits
                
        feasibility_rate = (feasible_shots / total_shots) * 100
        
        print(f"{sc:<26} | {noisy_energy:<22.4f} | {feasibility_rate:>15.2f}%")
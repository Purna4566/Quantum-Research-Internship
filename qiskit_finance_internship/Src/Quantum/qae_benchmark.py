import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler as Sampler
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. THE MARKET ENVIRONMENT
# ==========================================
# 8-scenario synthetic market (Losses $0M to $7M)
MARKET_PROBS = [0.05, 0.15, 0.30, 0.25, 0.15, 0.05, 0.04, 0.01]
VAR_THRESHOLD = 6 # 95% VaR is >= $6M
TRUE_TAIL_PROB = sum(MARKET_PROBS[VAR_THRESHOLD:]) # Exact answer = 0.05

# ==========================================
# 2. CLASSICAL MONTE CARLO (WALL STREET STANDARD)
# ==========================================
def run_classical_monte_carlo(num_samples):
    start_time = time.time()
    
    # Simulate drawing random market scenarios from the probability distribution
    simulated_scenarios = np.random.choice(
        range(len(MARKET_PROBS)), 
        size=num_samples, 
        p=MARKET_PROBS
    )
    
    # Count how many times the portfolio crashed past our VaR threshold
    tail_events = np.sum(simulated_scenarios >= VAR_THRESHOLD)
    estimated_prob = tail_events / num_samples
    
    runtime = time.time() - start_time
    accuracy = 100 - (abs(estimated_prob - TRUE_TAIL_PROB) / TRUE_TAIL_PROB) * 100
    
    return estimated_prob, num_samples, runtime, accuracy

# ==========================================
# 3. QUANTUM AMPLITUDE ESTIMATION 
# ==========================================
def build_quantum_problem():
    amplitudes = np.sqrt(MARKET_PROBS)
    
    # 1. State Prep (Data Qubits)
    state_prep = QuantumCircuit(4)
    state_prep.prepare_state(amplitudes, [0, 1, 2])
    
    # 2. Comparator Logic (Flag the target VaR scenarios)
    for state_val in range(VAR_THRESHOLD, 8):
        bin_str = format(state_val, '03b')
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0': state_prep.x(i)
        
        state_prep.mcx([0, 1, 2], 3)
        
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0': state_prep.x(i)
            
    return EstimationProblem(state_preparation=state_prep, objective_qubits=[3])

def run_quantum_iqae(epsilon_target):
    problem = build_quantum_problem()
    sampler = Sampler()
    iqae = IterativeAmplitudeEstimation(epsilon_target=epsilon_target, alpha=0.05, sampler=sampler)
    
    start_time = time.time()
    result = iqae.estimate(problem)
    runtime = time.time() - start_time
    
    estimated_prob = result.estimation
    queries = result.num_oracle_queries
    accuracy = 100 - (abs(estimated_prob - TRUE_TAIL_PROB) / TRUE_TAIL_PROB) * 100
    
    return estimated_prob, queries, runtime, accuracy

# ==========================================
# 4. EXECUTION & DASHBOARD GENERATION
# ==========================================
if __name__ == "__main__":
    print("=========================================================================")
    print("      TASK 4: QUANTUM VS CLASSICAL RISK BENCHMARKING DASHBOARD           ")
    print("=========================================================================\n")
    print(f"Target: 95% Value-at-Risk (Threshold >= ${VAR_THRESHOLD}M) | True Probability: {TRUE_TAIL_PROB:.4f}\n")
    
    print("--- CLASSICAL MONTE CARLO ---")
    print(f"{'Samples (M)':<12} | {'Est. Prob':<10} | {'Accuracy':<10} | {'Runtime'}")
    print("-" * 55)
    
    # Run Classical for 10^3, 10^4, 10^5, and 10^6 samples
    classical_samples = [1000, 10000, 100000, 1000000]
    for m in classical_samples:
        est_p, queries, rt, acc = run_classical_monte_carlo(m)
        print(f"{queries:<12} | {est_p:<10.4f} | {acc:>7.2f}%   | {rt:.4f}s")
        
    print("\n--- QUANTUM AMPLITUDE ESTIMATION (IQAE) ---")
    print(f"{'Error Target':<12} | {'Est. Prob':<10} | {'Accuracy':<10} | {'Grover Queries':<15} | {'Runtime'}")
    print("-" * 75)
    
    # Run Quantum for 5%, 1%, and 0.5% error targets
    quantum_epsilons = [0.05, 0.01, 0.005]
    for eps in quantum_epsilons:
        est_p, queries, rt, acc = run_quantum_iqae(eps)
        print(f"Eps = {eps:<6} | {est_p:<10.4f} | {acc:>7.2f}%   | {queries:<15} | {rt:.4f}s")
        
    print("\n=========================================================================")
import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler as Sampler
import warnings

# Suppress deprecation warnings for cleaner console output
warnings.filterwarnings("ignore")

def build_portfolio_distribution():
    """
    Creates a 3-qubit state representing 8 possible portfolio loss scenarios 
    (from $0M to $7M). The probabilities are tailored to have specific tail 
    risks for our 90%, 95%, and 99% VaR confidence intervals.
    """
    # Probabilities for losses: $0M, $1M, $2M, $3M, $4M, $5M, $6M, $7M
    # Notice how the extreme losses (tail end) have very small probabilities.
    probabilities = [0.05, 0.15, 0.30, 0.25, 0.15, 0.05, 0.04, 0.01]
    
    # Qiskit requires amplitudes (square root of probabilities)
    amplitudes = np.sqrt(probabilities)
    
    prep_circ = QuantumCircuit(3)
    prep_circ.prepare_state(amplitudes, [0, 1, 2])
    return prep_circ

def build_comparator(threshold):
    """
    Builds a Boolean logic oracle. It reads the 3-qubit loss state and 
    flips a 4th 'objective' qubit to |1> if the loss is >= the threshold.
    """
    circ = QuantumCircuit(4)
    
    # Iterate through all loss states that exceed our threshold
    for state_val in range(threshold, 8):
        # Convert integer to a 3-bit binary string (e.g., 5 -> '101')
        bin_str = format(state_val, '03b')
        
        # Qiskit is little-endian (qubit 0 is the rightmost bit).
        # We apply X-gates to target the '0' bits so our Multi-Controlled X (MCX) fires correctly.
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0':
                circ.x(i)
                
        # Multi-Controlled X: If data qubits match the state, flip the objective qubit (qubit 3)
        circ.mcx([0, 1, 2], 3)
        
        # Uncompute: Revert the X-gates to preserve the original quantum state
        for i, bit in enumerate(reversed(bin_str)):
            if bit == '0':
                circ.x(i)
                
    return circ

def evaluate_var_threshold(threshold, expected_tail_prob):
    """
    Assembles the state preparation and comparator into an EstimationProblem, 
    then runs Iterative QAE to calculate the VaR tail probability.
    """
    # 1. State Preparation (Data Qubits)
    data_prep = build_portfolio_distribution()
    
    # 2. Comparator (Logic Qubit)
    comparator = build_comparator(threshold)
    
    # 3. Combine into a single 4-qubit circuit
    state_prep = QuantumCircuit(4)
    state_prep.compose(data_prep, qubits=[0,1,2], inplace=True)
    state_prep.compose(comparator, qubits=[0,1,2,3], inplace=True)
    
    # 4. Configure IQAE
    problem = EstimationProblem(
        state_preparation=state_prep,
        objective_qubits=[3]
    )
    
    sampler = Sampler()
    # Setting an aggressive epsilon to force accurate tail measurements
    iqae = IterativeAmplitudeEstimation(epsilon_target=0.005, alpha=0.05, sampler=sampler)
    
    # 5. Execute and track runtime
    start_time = time.time()
    result = iqae.estimate(problem)
    runtime = time.time() - start_time
    
    # 6. Performance Analytics
    est_prob = result.estimation
    queries = result.num_oracle_queries
    accuracy = 100 - (abs(est_prob - expected_tail_prob) / expected_tail_prob) * 100
    
    return est_prob, runtime, queries, accuracy

if __name__ == "__main__":
    print("=== QUANTUM VALUE-AT-RISK (VaR) ESTIMATION ===")
    print("Asset Universe: Synthetic 8-Scenario Portfolio")
    
    # Define our VaR targets based on the programmed distribution.
    # To find the 90% VaR, we look for the threshold where the tail probability is 10% (0.10).
    var_targets = [
        {"conf": "90%", "threshold": 5, "expected": 0.10},  # Sum of P(5,6,7) = 0.05 + 0.04 + 0.01 = 0.10
        {"conf": "95%", "threshold": 6, "expected": 0.05},  # Sum of P(6,7) = 0.04 + 0.01 = 0.05
        {"conf": "99%", "threshold": 7, "expected": 0.01}   # Sum of P(7) = 0.01
    ]
    
    print("\n" + "-" * 85)
    print(f"{'VaR Conf':<10} | {'Threshold':<12} | {'True Tail P':<12} | {'Est P':<8} | {'Accuracy':<10} | {'Queries':<8} | {'Runtime'}")
    print("-" * 85)
    
    for target in var_targets:
        est_p, runtime, queries, acc = evaluate_var_threshold(target["threshold"], target["expected"])
        print(f"{target['conf']:<10} | > ${target['threshold']} Million | {target['expected']:<12.4f} | {est_p:<8.4f} | {acc:>7.2f}%   | {queries:<8} | {runtime:.2f}s")
        
    print("-" * 85)
    print("\n[Task 3 Complete] Quantum VaR Analysis Pipeline executed successfully.")
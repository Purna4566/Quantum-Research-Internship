import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler as Sampler
import warnings

# Suppress deprecation warnings for cleaner console output
warnings.filterwarnings("ignore")

class QAEPipeline:
    """
    A modular Quantum Amplitude Estimation pipeline for financial probability analysis.
    """
    def __init__(self, target_probability):
        self.target_probability = target_probability
        self.encoder_circuit = None
        self.grover_op = None
        self.estimation_problem = None
        
    # ==========================================
    # MODULE 1: Probability Distribution Encoder
    # ==========================================
    def build_distribution_encoder(self):
        """
        Encodes a synthetic financial probability into the amplitude of a quantum state.
        For VaR, this represents the probability of a portfolio loss exceeding a threshold.
        """
        # Calculate the rotation angle theta required to encode the probability 'p'
        # Math: p = sin^2(theta/2)
        theta = 2 * np.arcsin(np.sqrt(self.target_probability))
        
        self.encoder_circuit = QuantumCircuit(1)
        self.encoder_circuit.ry(theta, 0)
        
        print(f"[Module 1] Distribution Encoder Built. Encoded P = {self.target_probability}")
        return self.encoder_circuit

    # ==========================================
    # MODULE 2: Grover Operator Construction
    # ==========================================
    def build_grover_operator(self):
        """
        Constructs the Q operator, performing reflections about the target state 
        and the mean to amplify the encoded probability.
        """
        if self.encoder_circuit is None:
            raise ValueError("Must build encoder circuit first.")
            
        # The objective is state |1>, so the oracle is simply an identity mapping on the qubit structure
        # Qiskit's GroverOperator handles the S_0 and S_chi reflections automatically.
        self.grover_op = GroverOperator(
            oracle=QuantumCircuit(1), 
            state_preparation=self.encoder_circuit, 
            reflection_qubits=[0]
        )
        print("[Module 2] Grover Operator Constructed.")
        return self.grover_op

    # ==========================================
    # MODULE 3: Amplitude Estimation Circuit
    # ==========================================
    def configure_estimation_problem(self):
        """
        Packages the state preparation and objective qubits into a Qiskit EstimationProblem.
        """
        self.estimation_problem = EstimationProblem(
            state_preparation=self.encoder_circuit,
            objective_qubits=[0]
        )
        print("[Module 3] Estimation Problem Configured.")
        return self.estimation_problem

    # ==========================================
    # MODULE 4: Measurement Pipeline
    # ==========================================
    def execute_measurement_pipeline(self, epsilon=0.01, alpha=0.05):
        """
        Executes Iterative QAE (IQAE) using the Qiskit Sampler primitive.
        - epsilon: Target precision (error margin).
        - alpha: Confidence level (alpha=0.05 means 95% confidence interval).
        """
        print(f"[Module 4] Executing IQAE Measurement Pipeline (Target Error: {epsilon*100}%, Confidence: {(1-alpha)*100}%)...")
        
        # We use Sampler V1 here as it is the standard for qiskit-algorithms IQAE
        sampler = Sampler()
        
        iqae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha,
            sampler=sampler
        )
       
        result = iqae.estimate(self.estimation_problem)
        return result

    # ==========================================
    # MODULE 5: Probability Extraction
    # ==========================================
    def extract_probabilities(self, result):
        """
        Parses the raw IQAE result object into human-readable financial metrics.
        """
        print("\n" + "="*40)
        print("[Module 5] PROBABILITY EXTRACTION REPORT")
        print("="*40)
        print(f"Target (True) Probability:    {self.target_probability:.4f}")
        print(f"Estimated Probability:        {result.estimation:.4f}")
        print(f"95% Confidence Interval:      [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        print(f"Total Grover Queries (Cost):  {result.num_oracle_queries}")
        print("="*40)


if __name__ == "__main__":
    # Experiment 1: Synthetic probability distribution mapping
    # Let's assume a 15% probability of a portfolio crash
    synthetic_loss_probability = 0.15 
    
    qae = QAEPipeline(target_probability=synthetic_loss_probability)
 
    qae.build_distribution_encoder()
    qae.build_grover_operator()
    qae.configure_estimation_problem()
    
    qae_result = qae.execute_measurement_pipeline(epsilon=0.01, alpha=0.05)
   
    qae.extract_probabilities(qae_result)
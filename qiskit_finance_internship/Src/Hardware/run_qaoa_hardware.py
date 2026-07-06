import numpy as np
import warnings
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# 1. AUTHENTICATION & TARGET HARDWARE
# ==========================================
API_TOKEN = "USIv7bsULuDUwa5n7OEVvvmXK-dJ4d7toNg7ZZer8-u7"
TARGET_BACKEND = "ibm_kingston" 

print(f"Connecting to IBM Quantum Platform... Target: {TARGET_BACKEND}")

# Direct authentication to bypass local cache
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_TOKEN)
backend = service.backend(TARGET_BACKEND)

# ==========================================
# 2. DEFINING THE 5-ASSET PORTFOLIO HAMILTONIAN
# ==========================================
# A scaled-down Ising Hamiltonian for a 5-asset portfolio matrix
hamiltonian = SparsePauliOp.from_list([
    ("ZIIII", 0.15), ("IZIII", -0.22), ("IIZII", 0.31), ("IIIZI", -0.05), ("IIIIZ", 0.12),
    ("ZZIII", 0.05), ("IZZII", 0.11), ("IIZZI", 0.08), ("IIIZZ", 0.14)
])

# ==========================================
# 3. CONSTRUCTING THE QAOA CIRCUIT (p=1)
# ==========================================
print("Constructing parameterized QAOA Ansatz (Depth p=1)...")
ansatz = QAOAAnsatz(hamiltonian, reps=1)

# Fix parameters (beta, gamma) to pre-optimized variational angles
angles = [0.423, 0.781] 
qaoa_circuit = ansatz.assign_parameters(angles)
qaoa_circuit.measure_all()

# ==========================================
# 4. HARDWARE TRANSPILATION
# ==========================================
print(f"Transpiling circuit for the native 156-qubit topology of {TARGET_BACKEND}...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
transpiled_circuit = pm.run(qaoa_circuit)

# ==========================================
# 5. SUBMITTING THE REAL QUANTUM JOB
# ==========================================
print(f"Submitting job to the cloud. Shot Count = 4096...")
sampler = Sampler(mode=backend)

# Submit to the IBM Quantum Runtime cluster
job = sampler.run([transpiled_circuit], shots=4096)

print("\n" + "="*50)
print("             JOB SUBMISSION REPORT            ")
print("="*50)
print(f"Successfully sent to: {TARGET_BACKEND}")
print(f"Your Cloud Job ID   : {job.job_id()}")
print("="*50)
print("\n[Action Required]: Copy your Job ID. You will need it to retrieve your real quantum financial results once it finishes the cloud queue!")
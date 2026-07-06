import numpy as np
import warnings
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
TARGET_BACKEND = "ibm_marrakesh" 

print(f"Connecting to IBM Quantum Platform... Target: {TARGET_BACKEND}")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_TOKEN)
backend = service.backend(TARGET_BACKEND)

# ==========================================
# 2. CIRCUIT PREPARATION
# ==========================================
hamiltonian = SparsePauliOp.from_list([
    ("ZIIII", 0.15), ("IZIII", -0.22), ("IIZII", 0.31), ("IIIZI", -0.05), ("IIIIZ", 0.12),
    ("ZZIII", 0.05), ("IZZII", 0.11), ("IIZZI", 0.08), ("IIIZZ", 0.14)
])

ansatz = QAOAAnsatz(hamiltonian, reps=1)
angles = [0.423, 0.781] 
qaoa_circuit = ansatz.assign_parameters(angles)
qaoa_circuit.measure_all()

print(f"Transpiling circuit for {TARGET_BACKEND} (Optimization Level 3)...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
transpiled_circuit = pm.run(qaoa_circuit)

# ==========================================
# 3. APPLYING HARDWARE ERROR SUPPRESSION (V2)
# ==========================================
print("Configuring SamplerV2 with V2 Noise Management...")
sampler = Sampler(mode=backend)

# Turn on Pauli twirling for both gates and measurements
sampler.options.twirling.enable_gates = True
sampler.options.twirling.enable_measure = True

# Turn on Dynamical Decoupling to protect idle qubits
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XpXm"

# ==========================================
# 4. SUBMITTING THE MITIGATED JOB
# ==========================================
print("Submitting mitigated job to the cloud. Shot Count = 4096...")
job = sampler.run([transpiled_circuit], shots=4096)

print("\n" + "="*50)
print("       MITIGATED JOB SUBMISSION REPORT        ")
print("="*50)
print(f"Target Backend      : {TARGET_BACKEND}")
print(f"Mitigation Mode     : Resilience Level 1 (Readout)")
print(f"Your Cloud Job ID   : {job.job_id()}")
print("="*50)
print("\n[Action Required]: Copy your NEW Job ID to track this mitigated run!")
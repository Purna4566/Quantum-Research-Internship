import numpy as np
import warnings
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

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
# 2. DEFINING THE MOCK HAMILTONIANS (5, 7, and 10 Assets)
# ==========================================
# 5 Assets
ham_5 = SparsePauliOp.from_list([
    ("ZIIII", 0.15), ("IZIII", -0.22), ("IIZII", 0.31), 
    ("ZZIII", 0.05), ("IZZII", 0.11)
])

# 7 Assets
ham_7 = SparsePauliOp.from_list([
    ("ZIIIIII", 0.12), ("IZIIIII", -0.15), ("IIZIIII", 0.22), ("IIIIIIZ", 0.18),
    ("ZZIIIII", 0.07), ("IZZIIII", 0.09), ("IIZZIII", 0.04)
])

# 10 Assets
ham_10 = SparsePauliOp.from_list([
    ("ZIIIIIIIII", 0.10), ("IZIIIIIIII", -0.12), ("IIZIIIIIII", 0.15), 
    ("IIIIIIIIIZ", 0.20), ("ZZIIIIIIII", 0.05), ("IZZIIIIIII", 0.08)
])

portfolios = [("5-Asset", ham_5), ("7-Asset", ham_7), ("10-Asset", ham_10)]

# Pre-optimized angles for p=1 (2 angles) and p=2 (4 angles)
angles_p1 = [0.423, 0.781]
angles_p2 = [0.312, 0.543, 0.612, 0.892]
depths = [(1, angles_p1), (2, angles_p2)]

# ==========================================
# 3. CONSTRUCTING & TRANSPILING ALL 6 CIRCUITS
# ==========================================
print("Constructing and transpiling QAOA circuits for all sizes and depths...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

transpiled_circuits = []
circuit_labels = []

for name, ham in portfolios:
    for p, angles in depths:
        # Build circuit
        ansatz = QAOAAnsatz(ham, reps=p)
        qc = ansatz.assign_parameters(angles)
        qc.measure_all()
        
        # Transpile
        t_qc = pm.run(qc)
        transpiled_circuits.append(t_qc)
        circuit_labels.append(f"{name} (p={p})")
        print(f" -> Prepared {name} at depth p={p}")

# ==========================================
# 4. BATCH SUBMISSION TO HARDWARE
# ==========================================
print(f"\nSubmitting batch job of 6 circuits to {TARGET_BACKEND}...")
sampler = Sampler(mode=backend)

# We will keep V2 error suppression enabled to ensure clean comparisons
sampler.options.twirling.enable_gates = True
sampler.options.twirling.enable_measure = True
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XpXm"

# Submit all circuits in a single batch
job = sampler.run(transpiled_circuits, shots=4096)

print("\n" + "="*50)
print("       BATCH SCALING JOB SUBMISSION REPORT        ")
print("="*50)
print(f"Target Backend      : {TARGET_BACKEND}")
print(f"Total Circuits      : {len(transpiled_circuits)}")
print(f"Your Cloud Job ID   : {job.job_id()}")
print("="*50)
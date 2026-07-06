from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
import numpy as np

# Same 5-asset Hamiltonian
ham = SparsePauliOp.from_list([
    ("ZIIII", 0.15), ("IZIII", -0.22), ("IIZII", 0.31), 
    ("ZZIII", 0.05), ("IZZII", 0.11)
])

# Simulate perfectly
ansatz = QAOAAnsatz(ham, reps=1)
qc = ansatz.assign_parameters([0.423, 0.781])
qc.measure_all()

# Use Statevector (Noiseless)
sampler = StatevectorSampler()
job = sampler.run([qc], shots=4096)
counts = job.result()[0].data.meas.get_counts()

print("--- SIMULATOR RESULTS (GROUND TRUTH) ---")
for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"|{state}> : {(count/4096)*100:.1f}%")
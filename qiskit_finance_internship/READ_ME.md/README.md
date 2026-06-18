# Quantum Finance Research Internship

This repository contains the foundational code and documentation for my Quantum Algorithm Research Internship, focusing on quantum circuit fundamentals and the Quantum Approximate Optimization Algorithm (QAOA) for financial applications.

## 1. Quantum Basics
Unlike classical computers that use bits (0 or 1), quantum computers leverage **Qubits**. A qubit can exist in a continuum of states until it is measured.

* **Superposition:** The ability of a qubit to exist in a linear combination of both |0> and |1> states simultaneously.
* **Entanglement:** A uniquely quantum phenomenon where two or more qubits become perfectly correlated. Measuring one instantly determines the state of the other, regardless of physical distance.
* **The Optimization Principle:** Quantum optimization algorithms work by mapping a complex computational problem to a physical system's energy landscape. The quantum computer searches for the **Ground State** (the lowest energy configuration), which corresponds to the optimal mathematical solution.

## 2. Fundamental Quantum Gates
Quantum gates are the operational building blocks of quantum circuits. 

* **Hadamard Gate (H):** Acts on a single qubit to create a perfect 50/50 superposition.
* **Pauli-X Gate:** The quantum equivalent of a classical NOT gate. It flips a qubit from |0> to |1>.
* **Pauli-Z Gate:** Introduces a phase flip, shifting the phase of the |1> state by pi radians.
* **Controlled-NOT Gate (CNOT / CX):** A two-qubit gate that flips the target qubit if and only if the control qubit is in the |1> state. This is the primary tool used to generate entanglement.

## 3. The Bell State
The Bell State represents a maximally entangled two-qubit system. By applying a Hadamard gate followed by a CNOT gate, the qubits are locked into a state where they will always yield identical measurements (either 00 or 11, roughly 50% of the time each).

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts()
print(f"Measurement Counts: {counts}")  
qc.draw(output='text')

## 4. Introduction to QAOA
The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical variational algorithm designed to solve combinatorial optimization problems, such as portfolio optimization.

The Workflow:

Mathematical Mapping: The business optimization problem is translated into a Quadratic Program (QUBO) containing binary variables.

Problem Hamiltonian (H_c): The cost function of the problem is encoded into a physical energy landscape.

Mixer Hamiltonian (H_b): An operator that drives quantum transitions to ensure the algorithm searches across the entire solution landscape without getting stuck in local minima.

The Variational Loop: The quantum circuit applies H_c and H_b alternately using parameterized angles. A classical optimizer (like COBYLA) reads the measurement outputs, adjusts the angles, and feeds them back into the quantum circuit until the optimal solution is found.

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

def setup_ibm_runtime():
    """
    Authenticates and connects to the IBM Quantum Runtime service directly.
    """
    print("Authenticating with IBM Quantum Cloud...")
    
    # 👉 PUT YOUR BRAND NEW API TOKEN INSIDE THESE QUOTES
    NEW_API_TOKEN = "USIv7bsULuDUwa5n7OEVvvmXK-dJ4d7toNg7ZZer8-u7"
    
    # Initialize directly to bypass any broken local save files
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", 
        token=NEW_API_TOKEN
    )
    
    print("Authentication Successful! Fetching hardware metrics...\n")
    return service

def analyze_backend(service, backend_name):
    """
    Pings a specific IBM QPU and extracts real-time queue times,
    qubit counts, and median gate error metrics.
    """
    try:
        backend = service.backend(backend_name)
        status = backend.status()
        properties = backend.properties()
        
        num_qubits = backend.num_qubits
        pending_jobs = status.pending_jobs
        is_operational = status.operational
        
        if properties:
            # 1. Readout Error (Measurement fidelity)
            readout_errors = []
            for qubit in properties.qubits:
                for nduv in qubit:
                    if nduv.name == 'readout_error':
                        readout_errors.append(nduv.value)
            
            med_readout = np.median(readout_errors) if readout_errors else 0.0
            
            # 2. CX Error (Two-qubit gate fidelity)
            cx_errors = []
            for gate in properties.gates:
                if len(gate.qubits) == 2:
                    for param in gate.parameters:
                        if param.name == 'gate_error':
                            cx_errors.append(param.value)
                            
            med_cx_error = np.median(cx_errors) if cx_errors else 0.0
        else:
            med_readout = 0.0
            med_cx_error = 0.0
        
        return {
            "name": backend_name,
            "operational": is_operational,
            "qubits": num_qubits,
            "pending_jobs": pending_jobs,
            "median_readout_error": med_readout,
            "median_cx_error": med_cx_error
        }
    except Exception as e:
        return {"name": backend_name, "error": str(e)}

if __name__ == "__main__":
    print("=== IBM QUANTUM HARDWARE BENCHMARK ===")
    
    # Initialize connection
    service = setup_ibm_runtime()
    
    # Dynamically fetch all real, operational quantum hardware available to your specific account
    available_backends = service.backends(simulator=False, operational=True)
    target_backends = [backend.name for backend in available_backends]
    
    # Generate the Comparison Dashboard
    print("-" * 90)
    print(f"{'Backend Name':<15} | {'Status':<8} | {'Qubits':<8} | {'Queue (Jobs)':<14} | {'Med Readout Err':<17} | {'Med CX Err'}")
    print("-" * 90)
    
    for name in target_backends:
        data = analyze_backend(service, name)
        
        if "error" in data:
            print(f"{data['name']:<15} | ERROR: {data['error']}")
        else:
            status_str = "Active" if data["operational"] else "Offline"
            readout_str = f"{data['median_readout_error']:.4f}"
            cx_str = f"{data['median_cx_error']:.4f}"
            
            print(f"{data['name']:<15} | {status_str:<8} | {data['qubits']:<8} | {data['pending_jobs']:<14} | {readout_str:<17} | {cx_str}")
            
    print("-" * 90)
    print("\n[Task 1 Complete] Hardware telemetry captured successfully.")
import warnings
from qiskit_ibm_runtime import QiskitRuntimeService

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# 1. AUTHENTICATION & TARGET JOB
# ==========================================
API_TOKEN = "USIv7bsULuDUwa5n7OEVvvmXK-dJ4d7toNg7ZZer8-u7"
JOB_ID = "d94co97u62ks7395s6b0"

print(f"Connecting to IBM Quantum Cloud...")
print(f"Pinging Job ID: {JOB_ID}...")

# Authenticate session
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_TOKEN)

# ==========================================
# 2. STATUS CHECK & DATA EXTRACTION
# ==========================================
try:
    job = service.job(JOB_ID)
    status = job.status()
    
    print("\n" + "="*45)
    print(f" JOB STATUS: {status}")
    print("="*45)

    if status == "DONE":
        print("Job is complete! Downloading results from the QPU...")
        result = job.result()
        
        # SamplerV2 returns data in a PubResult format. 
        # 'meas' is the default name of the classical register from measure_all()
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()
        
        # Sort the measured quantum states by highest frequency (probability)
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        
        print("\n" + "-"*45)
        print(f" {'PORTFOLIO STATE':<20} | {'MEASUREMENT COUNT'} ")
        print("-"*45)
        
        # Display the top 10 most frequently measured states
        for state, count in sorted_counts[:10]:
            # Calculate probability percentage
            prob = (count / 4096) * 100 
            print(f" |{state}> {'':<11} | {count:<5} ({prob:.1f}%)")
            
        print("-"*45)
        print("\n[Task 2 Complete] Real hardware results successfully retrieved!")
        
    elif status in ["QUEUED", "RUNNING"]:
        print("\nThe QPU is still processing the queue.")
        print("Feel free to close this terminal. Your job is safe in the cloud.")
        print("Run this script again later to check back!")
    else:
        print("\nJob encountered an error or was cancelled. Check your IBM dashboard for details.")
        
except Exception as e:
    print(f"\nError retrieving job: {e}")
import matplotlib.pyplot as plt

# Shriya's JSON Data
qaoa_history = [
    0.0, 1.07e-16, 1.38e-17, 0.2338, 9.36e-17, 0.0176, -0.0007, -0.0257, -0.0681, 
    -0.3296, -1.6366, -0.9934, 0.0297, -1.8481, -1.9438, -1.8246, -1.6270, 
    -1.9251, -1.9966, -1.9780, -1.9910, -1.9914, -1.9972, -1.9982, -1.9989, 
    -1.9985, -1.9992, -1.9993, -1.9996, -1.9997, -1.9998, -1.9999, -1.9999, 
    -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, 
    -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999, -1.9999
]

vqe_history = [
    2.0, 1.5403, 1.0806, 0.9564, 0.8322, 0.3729, 0.4634, 0.1517, 0.3575, -0.0180, 
    -0.2461, -0.5206, -0.9109, -0.9280, -0.7629, -0.9664, -0.6463, -1.0653, -0.7816, 
    -1.2128, -1.1248, -0.7422, -1.3671, -1.7464, -1.4311, -1.5545, -1.7761, -1.5862, 
    -1.7397, -1.2753, -1.5781, -1.6216, -1.6895, -1.2124, -1.7515, -1.6428, -1.3069, 
    -1.5571, -1.7242, -1.7725, -1.7710, -1.7899, -1.7936, -1.7752, -1.7859, -1.8220, 
    -1.8235, -1.8147, -1.8109, -1.8745
]

# Plotting
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')

plt.plot(range(len(qaoa_history)), qaoa_history, label='QAOA', color='#00FFAA', linewidth=2)
plt.plot(range(len(vqe_history)), vqe_history, label='VQE', color='#FF00AA', linestyle='--', linewidth=2)

plt.axhline(y=-2.0, color='white', linestyle=':', alpha=0.5, label='Global Minimum')

plt.title("Joint Convergence Analysis: QAOA vs. VQE", fontsize=16, pad=15)
plt.xlabel("Optimization Iterations", fontsize=12)
plt.ylabel("Expectation Value (Energy)", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.1)

plt.savefig("joint_convergence_comparison.png", dpi=300)
plt.show()
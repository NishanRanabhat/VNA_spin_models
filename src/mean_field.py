import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_field_magnetization(J, T, system="FullyConnected", max_iter=100000, tol=1e-6):
    """
    Solves the mean-field equation for the unidirectional 1D Ising model.
    J: Interaction strength
    T: Temperature (array or scalar)
    max_iter: Maximum number of iterations for convergence
    tol: Convergence tolerance
    """
    m_values = []
    if system == "FullyConnected":
        for temp in np.atleast_1d(T):
            if temp == 0:
                m_values.append(1.0)  # At T=0, magnetization is fully ordered
                continue
        
            m = 1.0  # Initial guess
            for _ in range(max_iter):
                m_new = np.tanh(J * m / (temp))  # Mean-field equation (assuming k_B = 1)
                if np.abs(m_new - m) < tol:
                    break
                m = m_new
            m_values.append(m)
    
    return np.array(m_values)

def mean_field_local_energy(N, J, T_values, m):
    E_locs = []
    for i, T in enumerate(T_values):
        if T == 0:
            E_locs.append(-0.75*J*N*(m[i]**2))
        else:
            E_loc = -0.5*J*N*(m[i]**2)-N*J*m[i]*np.tanh(J*m[i]/T)
            E_locs.append(E_loc)
    return np.array(E_locs)


# Parameters
J = 1.0  # Interaction strength (assume J > 0 for ferromagnetic case)
T_values = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.5, 3.5]  # Temperature range
N_values = [50, 100, 200, 400, 800]  # System sizes

# Compute mean-field magnetization
m_values = mean_field_magnetization(J, T_values)

m2 = [mean_field_magnetization(J, T)**2 for T in T_values]


magnetizations = [[np.mean(np.load(f"output_files/check_finite_size/BGD_mag_temperature={T}_N={N}.npy")**2)
                  for T in T_values] for N in N_values]

# Create DataFrame and save to CSV

# Create dictionary of data
data = {'Temperature': T_values, 'Mean_Field': m2}
for i, N in enumerate(N_values):
    data[f'N={N}'] = magnetizations[i]

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('magnetizations.csv', index=False)

#magnetizations_l1u200 = [np.mean(np.load(f"output_files/layers_1_units_200/BGD_mag_temperature={T}_N=200.npy"))
#                             for T in T_values] 

#magnetizations_l3u200 = [np.mean(np.load(f"output_files/layers_3_units_200/BGD_mag_temperature={T}_N=200.npy"))
#                             for T in T_values] 

print("Magnetizations from BGD")
print(magnetizations[-1])
print("Magnetizations from Mean Field")
print(m2)

plt.figure(figsize=(10, 6))
for i, N in enumerate(N_values):
    plt.plot(T_values, magnetizations[i], label=f"N={N}", marker='o')
#plt.plot(T_values, magnetizations_l1u200, label="N=200 (1 layer)", marker='o')
#plt.plot(T_values, magnetizations_l3u200, label="N=200 (1 layer)", marker='o')
plt.plot(T_values, m2, label="Mean Field", color='black', linestyle='--')
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization")
plt.title("Finite Size Magnetization vs Temperature")
plt.legend()
plt.grid()
plt.savefig('finite_size_magnetization.pdf')







"""
for i, T in enumerate(T_values):
    print(f"T = {T}")
    print(f"Magnetization from .npy file: {np.mean(magnetizations[i])}")
    print(f"Magnetization from mean field: {m_2[i]}")
    print(f"Local Energy from .npy file: {np.mean(Elocs[i])}")
    print(f"Local Energy from mean field: {mean_field_local_energies[i]}")
    print(f"Free Energy from .npy file: {np.mean(varFlocs[i])}")
    print(f"Free Energy from mean field: {mean_field_free_energies[i]}")
    print(f"Log Probability from .npy file: {np.mean(log_probs[i])}")
    print(f"Log Probability from mean field: {mean_field_log_probs[i]}")
    print()
"""
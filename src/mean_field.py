import numpy as np
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
N_values = [20, 40, 60, 80, 100]  # System sizes

# Compute mean-field magnetization
m_values = mean_field_magnetization(J, T_values)

m2 = [mean_field_magnetization(J, T)**2 for T in T_values]


magnetizations = [[np.mean(np.load(f"output_files/mag_temperature={T}_N={N}.npy")**2)
                  for T in T_values] for N in N_values]


for i, N in enumerate(N_values):
    color = plt.cm.tab10(i % 10)
    plt.plot(T_values, magnetizations[i], "o-", label=f'N={N}', color=color)
plt.plot(T_values, m2, "o--", label="Mean field", color="red")
plt.xlabel('T')
plt.ylabel('$m^2$')
plt.legend(loc='upper right')
plt.savefig('figures/magnetizations.pdf')


"""
for N in N_values:
    color = plt.cm.tab10(N % 10)
    for i, T in enumerate(T_values):
        plt.plot(magnetizations[N_values.index(N)][i], label=f'T={T}, N={N}', color=color)

for i, T in enumerate(T_values):
    color = plt.cm.tab10(i % 10)
    plt.plot(magnetizations[i], label=f'T={T}, N={N}', color=color)
    plt.axhline(y=m_2[i], linestyle='--', color=color)
    plt.xlabel('epoch')
    plt.ylabel('$m^2$')
    plt.legend(loc='upper right')
plt.savefig('figures/magnetizations.png')
plt.close()
"""




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
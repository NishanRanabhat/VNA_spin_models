import numpy as np
from scipy.optimize import fsolve

def solve_m(J, T, initial_guess=0.5):
    # Define the equation m = tanh(J*m/T)
    func = lambda m: m - np.tanh(J * m / T)
    # Use fsolve to find the root
    m_solution, = fsolve(func, initial_guess)
    return m_solution

def energy_per_spin(m, J):
    return 0.5 * J * m**2

def free_energy(m, J, T):
    positive_coeff = 0.5*(1+m)
    negative_coeff = 0.5*(1-m)
    Eloc = 0.5*J*m**2
    return -Eloc + T*((0.5*(1+m))*np.log(0.5*(1+m)) + (0.5*(1-m))*np.log(0.5*(1-m)))

if __name__ == "__main__":
    # Example usage:
    J = 1.0
    T = 1.0
    m = solve_m(J, T)
    print(f"Solution for m: {m}")
    print(f"Corresponding free energy: {free_energy(m, J, T)}")


import numpy as np
import matplotlib.pyplot as plt
from mean_field import solve_m
import os


def load_data(num_sample, N, Tc, layer, Nh, taus, 
                        observable='squared_magnetization'):
    """
    Load data from dynamical FSS simulations for different annealing times
    (inversed annealing speed).
    """
    obs = np.empty(len(taus), dtype=np.float32)
    errors = np.empty(len(taus), dtype=np.float32)
    for i in range(len(taus)):
        dir_ = (f'../output_files/dynamical_fss/'
                f'sample{num_sample}_layer{layer}_Nh{Nh}_tau{taus[i]}/')
        if observable == 'squared_magnetization':
            filename = f'VNAtrain_mag2_temperature={Tc}_N={N}.npy'
            var_filename=f'VNAtrain_varmag2_temperature={Tc}_N={N}.npy'
        elif observable == 'magnetization':
            filename = f'VNAtrain_mag_temperature={Tc}_N={N}.npy'
            var_filename=f'VNAtrain_varmag_temperature={Tc}_N={N}.npy'
        elif observable == 'tesseractic_magnetization':
            filename = f'VNAtrain_mag4_temperature={Tc}_N={N}.npy'
            var_filename=f'VNAtrain_varmag4_temperature={Tc}_N={N}.npy'
        elif observable == 'free_energy':
            filename = f'VNAtrain_Floc_temperature={Tc}_N={N}.npy'
            var_filename=f'VNAtrain_varFloc_temperature={Tc}_N={N}.npy'
        data = np.load(os.path.join(dir_, filename))
        var_data = np.load(os.path.join(dir_, var_filename))
        # take only observable and variance at the target temperature
        obs[i] = data[-1] 
        errors[i] = np.sqrt(var_data[-1]) / np.sqrt(num_sample)
    return obs, errors

def create_dataset(num_sample, N_list, Tc, layer, Nh, taus, 
                   observable='squared_magnetization'):
    observables = np.empty(len(N_list))
    errors = np.empty(len(N_list))
    for i, N in enumerate(N_list):
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, 
                                observable)
        observables[i] = obs[-1]
        errors[i] = errors[-1]
    return np.array(N_list), taus, observables, errors

def compare_system_size(num_sample, Ns, Tc, layer, Nh, taus, 
                    observable='squared_magnetization'):

    plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    for N in Ns:
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, observable)
        plt.errorbar(taus, obs, yerr=errors, label=f'N={N}', fmt='o', capsize=4)

    plt.xlabel(r'$\tau$', fontsize=15)
    if observable == 'squared_magnetization':
        plt.ylabel(r'$\langle m^2 \rangle$', fontsize=15)
    elif observable == 'tesseractic_magnetization':
        plt.ylabel(r'$\langle m^4 \rangle$', fontsize=15)
    elif observable == 'free_energy':
        plt.ylabel(r'$\langle F_{\mathrm{loc}} \rangle$', fontsize=15)
    elif observable == 'magnetization':
        plt.ylabel(r'$\langle |m| \rangle$', fontsize=15)
    plt.legend()
    plt.grid()
    plt.savefig(f'../figures/adiabaticity_{observable}_layer{layer}_Nh{Nh}.pdf')
    plt.close()      

def compare_num_samples(num_samples, N, Tc, layer, Nh, taus, 
                        observable='squared_magnetization'):

    plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    for num_sample in num_samples:
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, observable)
        if num_sample >=256 and Nh>=256:
            num_sample = int(4*num_sample)
        plt.errorbar(taus, obs, yerr=errors, label=f'{num_sample} training samples', fmt='--o', capsize=4)

    plt.xlabel(r'$\tau$', fontsize=15)
    if observable == 'squared_magnetization':
        plt.ylabel(r'$\langle m^2 \rangle$', fontsize=15)
    elif observable == 'tesseractic_magnetization':
        plt.ylabel(r'$\langle m^4 \rangle$', fontsize=15)
    elif observable == 'free_energy':
        plt.ylabel(r'$\langle F_{\mathrm{loc}} \rangle$', fontsize=15)
    elif observable == 'magnetization':
        plt.ylabel(r'$\langle |m| \rangle$', fontsize=15)
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig(f'../figures/adiabaticity_{observable}_layer{layer}_Nh{Nh}_N{N}.pdf')
    plt.close()     

def compare_Nh(num_sample, N, Tc, layer, Nhs, taus, 
                        observable='squared_magnetization'):

    plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    for Nh in Nhs:
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, observable)
        plt.errorbar(taus, obs, yerr=errors, label=f'{Nh} hidden units', fmt='--o', capsize=4)

    plt.xlabel(r'$\tau$', fontsize=15)
    if observable == 'squared_magnetization':
        plt.ylabel(r'$\langle m^2 \rangle$', fontsize=15)
    elif observable == 'tesseractic_magnetization':
        plt.ylabel(r'$\langle m^4 \rangle$', fontsize=15)
    elif observable == 'free_energy':
        plt.ylabel(r'$\langle F_{\mathrm{loc}} \rangle$', fontsize=15)
    elif observable == 'magnetization':
        plt.ylabel(r'$\langle |m| \rangle$', fontsize=15)
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig(f'../figures/adiabaticity_{observable}_layer{layer}_samples{num_sample}_N{N}.pdf')
    plt.close()   

def compare_layers(num_sample, N, Tc, layers, Nh, taus, 
                        observable='squared_magnetization'):

    plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    for layer in layers:
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, observable)
        plt.errorbar(taus, obs, yerr=errors, label=f'{layer} layers', fmt='--o', capsize=4)

    plt.xlabel(r'$\tau$', fontsize=15)
    if observable == 'squared_magnetization':
        plt.ylabel(r'$\langle m^2 \rangle$', fontsize=15)
    elif observable == 'tesseractic_magnetization':
        plt.ylabel(r'$\langle m^4 \rangle$', fontsize=15)
    elif observable == 'free_energy':
        plt.ylabel(r'$\langle F_{\mathrm{loc}} \rangle$', fontsize=15)
    elif observable == 'magnetization':
        plt.ylabel(r'$\langle |m| \rangle$', fontsize=15)
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig(f'../figures/adiabaticity_{observable}_Nh{Nh}_samples{num_sample}_N{N}.pdf')
    plt.close() 

def compare_system_size(num_sample, Ns, Tc, layer, Nh, taus, 
                        observable='squared_magnetization'):

    plt.figure(figsize=(10, 7))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

    for N in Ns:
        obs, errors = load_data(num_sample, N, Tc, layer, Nh, taus, observable)
        plt.errorbar(taus, obs, yerr=errors, label=f'N={N}', fmt='--o', capsize=4)

    plt.xlabel(r'$\tau$', fontsize=15)
    if observable == 'squared_magnetization':
        plt.ylabel(r'$\langle m^2 \rangle$', fontsize=15)
    elif observable == 'tesseractic_magnetization':
        plt.ylabel(r'$\langle m^4 \rangle$', fontsize=15)
    elif observable == 'free_energy':
        plt.ylabel(r'$\langle F_{\mathrm{loc}} \rangle$', fontsize=15)
    elif observable == 'magnetization':
        plt.ylabel(r'$\langle |m| \rangle$', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.xscale('log')
    plt.grid()
    plt.savefig(f'../figures/adiabaticity_{observable}_layer_{layer}_Nh{Nh}_samples{num_sample}.pdf')
    plt.close() 

compare_system_size(num_sample=32768, 
                    Ns=[32, 64, 128, 256], 
                    Tc=1.0, layer=1, Nh=10, 
                    taus=[4, 5, 7, 8, 9, 16, 24, 32, 48, 64, 80, 96, 128,
                          196, 256, 324, 400, 512, 1024, 2000, 4000, 6000,
                          8000, 10000], 
                    observable='squared_magnetization')
import numpy as np 
from data_set import ScalingDataset
from dynamical_fss import FiniteSizeScaling
import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV, LeaveOneOut, RandomizedSearchCV
import json
from sys import path
path.append('../src')
from dynamical_fss_postprocessing import load_data


#prepare data
#each dataset should have a system size L, the domain x, the range y, and the error in y
#then use these to create a ScalingDataset object
num_sample=32768
Tc=1.0 
layer=1 
Nh=10
observable='squared_magnetization'
Ls = [32, 64, 128, 256]
x = np.array([48, 56, 64, 72, 80, 88, 96, 128,
              196, 256, 280, 324, 364, 400, 448, 512, 1024, 2000, 3000, 
              4000, 6000, 7000, 8000, 9000, 10000, 16000, 20000, 32000])
x_inv = 1/x

L1 = Ls[0]
y1, err1 = load_data(num_sample, L1, Tc, layer, Nh, x, observable)
ds1 = ScalingDataset(L1,x_inv,y1,err1) 

L2 = Ls[1]
y2, err2 = load_data(num_sample, L2, Tc, layer, Nh, x, observable)
ds2 = ScalingDataset(L2,x_inv,y2,err2) 

L3 = Ls[2]
y3, err3 = load_data(num_sample, L3, Tc, layer, Nh, x, observable)
ds3 = ScalingDataset(L3,x_inv,y3,err3) 

L4 = Ls[3]
y4, err4 = load_data(num_sample, L4, Tc, layer, Nh, x, observable)
ds4 = ScalingDataset(L4,x_inv,y4,err4) 

#initiate a FiniteSizeScaling object
s_factor = 1.0 # spline smoothing factor multiplier (default 1.0)
k = 3 # spline degree (default 3)
method = 'Nelder-Mead' #optimizer method for fitting (default 'Nelder-Mead')
maxiter = 100000 #maximum iterations for optimizer (default 1000)
a0_, b0_ = 1.5, 2.0 #initial guesses for scaling exponents a and b
fss = FiniteSizeScaling(ds1, ds2, ds3, ds4,
                        s_factor=s_factor, k=k,
                        method=method, maxiter=maxiter,
                        a0=a0_, b0=b0_)
### Fit spline using the FiniteSizeScaling clas
a_, b_ = fss.fit()
#spline, x_, y_ = fss.get_spline()
print(f"Fitted exponents: a = {a_}, b = {b_}")
"""
### Finding the optimal parameters using RandomizedSearchCV
param_grid = {
    's_factor': np.arange(1.0, 5.0, 0.1).tolist(),  # More conservative smoothing
    'k': [2, 3, 4, 5],  # Lower order splines are more stable
    'a0': np.linspace(0.1, 1.0, 10).tolist(),  # More reasonable initial guesses
    'b0': np.linspace(0.1, 1.0, 10).tolist(),
}
grid = RandomizedSearchCV(FSS_Estimator(maxiter=10000),
                    param_distributions=param_grid,
                    n_iter=1000,
                    cv=LeaveOneOut().split([ds1, ds2, ds3, ds4]))
grid.fit([ds1, ds2, ds3, ds4])
best_estimator = grid.best_estimator_
a_ = best_estimator.fss.a_
b_ = best_estimator.fss.b_
spline, x_, y_ = best_estimator.fss.get_spline()

# Convert data at each system size to log scale for plotting
x_s, y_s = best_estimator.fss.get_collapse_data()
cost = best_estimator.fss.cost

optimal_params = {"a": a_, "b": b_, "s_factor": best_estimator.s_factor,
                  "k": best_estimator.k, "cost": cost}
with open('./fully_connected_Ising_optimal_params.json', 'w') as f:
            json.dump(optimal_params, f, indent=4)
f.close()

print("Best parameters from grid search:")
print("s_factor =", best_estimator.s_factor)
print("k =", best_estimator.k)
print("a0 =", best_estimator.a0)
print("b0 =", best_estimator.b0)
print("FSS exponents: a =", a_, "b =", b_)
plt.figure()
for i in range(len(x_s)):
    plt.plot(x_s[i], y_s[i], 'o', label=f'L={int(fss.L_vals[i])}')
plt.plot(x_, spline(x_), label='Spline fit')

plt.xlabel(r'$log(tL^{-a})$')
plt.ylabel(r'$log(\langle m^2 \rangle L^{b})$')
plt.title('a = {:.4f}, b = {:.4f}, cost = {:.7f}'.format(a_, b_, cost))
plt.legend(loc='lower left')
plt.savefig('../figures/Fully_connected_Ising_spline.pdf')
plt.close()
"""
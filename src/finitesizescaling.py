import numpy as np
from scipy.optimize import curve_fit, minimize, dual_annealing, basinhopping 
from scipy import interpolate
from scipy.interpolate import interp2d, interp1d
from utilities import Y_rescaled,X_rescaled,slice_limits,closest_index

class FSS:

    def __init__(self,dataset,poly_order=8,initial_params=None,param_bounds=None,scaling_window=None,optimization_routine=None):

        self.dataset = dataset
        self.poly_order = poly_order
        self.initial_params = initial_params
        self.param_bounds = param_bounds
        self.scaling_window = scaling_window
        self.optimization_routine = optimization_routine

    def rescaled_combined_data(self,params):

        X_fin = np.asarray([])
        Y_fin = np.asarray([])
        
        for L_index,L in enumerate(self.dataset.system_size_list):
            X_rescale = X_rescaled(self.dataset.domain_list,L,params[0],params[2])
            Y_rescale = Y_rescaled(self.dataset.range_list[L_index,:],L,params[1])
            X_cut = slice_limits(X_rescale,self.scaling_window[0],self.scaling_window[1])
            Y_cut =  Y_rescale[closest_index(X_rescale,X_cut[0]):closest_index(X_rescale,X_cut[len(X_cut)-1])+1]
            
            X_fin = np.concatenate((X_fin,X_cut),axis=0)
            Y_fin = np.concatenate((Y_fin,Y_cut),axis=0)
        
        return X_fin,Y_fin

    def objective_function(self,params):
        X_fin,Y_fin = self.rescaled_combined_data(params)
        z = np.polyfit(X_fin, Y_fin,self.poly_order, full = True)
        return z[1]

    def optimization(self,):

        if self.optimization_routine == "L-BFGS-B":
            if self.param_bounds != None:
                result = minimize(self.objective_function, x0=self.initial_params, method=self.optimization_routine, bounds=self.param_bounds, options= {'ftol': 1e-12,'gtol': 1e-12,'maxiter': 10000})
            else:
                raise ValueError("L-BFGS-B need valid param-bounds")
        elif self.optimization_routine == "Nelder-Mead":
            result = minimize(self.objective_function, x0=self.initial_params, method=self.optimization_routine, options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
        elif self.optimization_routine == "dual-annealing":
            if self.param_bounds != None:
                result = dual_annealing(self.objective_function,bounds=self.param_bounds)
            else:
                raise ValueError("dual-annealing need valid param-bounds")

        return result.x, result.fun

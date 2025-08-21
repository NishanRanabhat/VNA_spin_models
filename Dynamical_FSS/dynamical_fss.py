import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

class FiniteSizeScaling:
    """
    Perform dynamical finite-size scaling analysis on a collection of ScalingDataset instances.

    Parameters:
        datasets: list of ScalingDataset
        s_factor: spline smoothing factor multiplier (default 1.0)
        k: spline degree (default 3)
        method: optimizer method for fitting (default 'Nelder-Mead')
        maxiter: maximum iterations for optimizer (default 1000)
        a0, b0: initial guesses for scaling exponents a and b
    """
    def __init__(self,
                 *datasets,
                 s_factor: float = 1.0,
                 k: int = 3,
                 method: str = 'Nelder-Mead',
                 maxiter: int = 1000,
                 a0: float = 1.0,
                 b0: float = 0.5):
        self.datasets = datasets
        self.s_factor = s_factor
        self.k = k
        self.method = method
        self.maxiter = maxiter
        self.a0 = a0
        self.b0 = b0
        self._prepare_data()

    def _prepare_data(self):
        """
        Flatten dataset lists into concatenated arrays for fitting.
        """
        self.L_vals = [ds.L for ds in self.datasets]
        self.x_list = [ds.x for ds in self.datasets]
        self.y_list = [ds.y for ds in self.datasets]
        self.err_list = [ds.err for ds in self.datasets]

    @staticmethod
    def _flatten(L_vals, x_list, y_list, err_list):
        L_rep = np.concatenate([
            np.full_like(x_list[i], L_vals[i], dtype=float)
            for i in range(len(L_vals))
        ])
        x_flat = np.concatenate(x_list)
        y_flat = np.concatenate(y_list)
        e_flat = np.concatenate(err_list)
        return L_rep, x_flat, y_flat, e_flat

    @staticmethod
    def _rescale(a: float, b: float,
                 L_rep: np.ndarray,
                 x_flat: np.ndarray,
                 y_flat: np.ndarray,
                 err_flat: np.ndarray,
                 eps: float = 1e-12):
        z = x_flat * L_rep ** (-a)
        y_s = y_flat * L_rep ** ( b)
        e_s = err_flat * L_rep ** ( b)

        z_log = np.log(np.maximum(z, eps))
        y_log = np.log(np.maximum(y_s, eps))
        sig_log = e_s / np.maximum(y_s, eps)
        mu, sd = y_log.mean(), y_log.std() + eps
        return z_log, (y_log - mu) / sd, sig_log / sd

    def _cost(self, params, L_rep, x_flat, y_flat, err_flat):
        a, b = params
        z, y_std, sig_std = self._rescale(a, b, L_rep, x_flat, y_flat, err_flat)
        idx = np.argsort(z)
        spl = UnivariateSpline(z[idx], y_std[idx], s=self.s_factor * len(z), k=self.k)
        self.cost = np.mean((y_std[idx] - spl(z[idx])) ** 2)
        return np.mean((y_std[idx] - spl(z[idx])) ** 2)

    def fit(self):
        """
        Optimize scaling exponents a, b.
        Returns:
            (a, b)
        """
        L_rep, x_flat, y_flat, err_flat = self._flatten(
            self.L_vals, self.x_list, self.y_list, self.err_list
        )
        options = {'maxiter': self.maxiter, 'xatol':1e-12, 'fatol':1e-12}
        res = minimize(
            self._cost,
            x0=(self.a0, self.b0),
            args=(L_rep, x_flat, y_flat, err_flat),
            method=self.method,
            options=options
        )
        self.a_, self.b_ = res.x
        return float(self.a_), float(self.b_)
    

    def get_spline(self) -> UnivariateSpline:
        """
        Return the fitted collapse spline after calling fit().
        """
        if not hasattr(self, 'a_') or not hasattr(self, 'b_'):
            raise RuntimeError('Call fit() before get_spline().')
        L_rep, x_flat, y_flat, err_flat = self._flatten(
            self.L_vals, self.x_list, self.y_list, self.err_list
        )

        z    = x_flat * L_rep**(-self.a_)
        ysc  = y_flat * L_rep**( self.b_)

        idx  = np.argsort(z)
        z_s  = z[idx]; y_s = ysc[idx]
        z_log= np.log(np.maximum(z_s,1e-12))
        y_log= np.log(np.maximum(y_s,1e-12))
        
        return UnivariateSpline(z_log, y_log, s=self.s_factor * len(z), k=self.k), z_log, y_log

    def get_collapse_data(self):
        """
        Return the fitted collapse spline after calling fit().
        """
        if not hasattr(self, 'a_') or not hasattr(self, 'b_'):
            raise RuntimeError('Call fit() before get_spline().')
        
        z_logs, y_logs = [], []

        for i in range(len(self.L_vals)):
            z = self.x_list[i] * self.L_vals[i]**(-self.a_)
            y = self.y_list[i] * self.L_vals[i]**(self.b_)
            idx  = np.argsort(z)
            z_s  = z[idx]; y_s = y[idx]
            z_log= np.log(np.maximum(z_s,1e-12))
            y_log= np.log(np.maximum(y_s,1e-12))
            z_logs.append(z_log)
            y_logs.append(y_log)
        return z_logs, y_logs        

    def score(self, x, y):
        """
        Cost function for scikit-learn's GridSearchCV or similar.
        """
        L_rep, x_flat, y_flat, err_flat = self._flatten(
        self.L_vals, self.x_list, self.y_list, self.err_list
        )
        # Use the fitted a_ and b_ values
        a, b = self.a_, self.b_
        z, y_std, sig_std = self._rescale(a, b, L_rep, x_flat, y_flat, err_flat)
        idx = np.argsort(z)
        spl = UnivariateSpline(z[idx], y_std[idx], s=self.s_factor * len(z), k=self.k)
        mse = np.mean((y_std[idx] - spl(z[idx])) ** 2)
         # Negative MSE, so higher for the scikit-learn model selection
         # (Grid search / Randomized search)
        return -mse 
    
class FSS_Estimator(BaseEstimator, RegressorMixin):
    """   Scikit-learn compatible estimator."""
    def __init__(self, s_factor=1.0, k=3, method='Nelder-Mead', maxiter=1000, a0=1.0, b0=0.5):
        self.s_factor = s_factor
        self.k = k
        self.method = method
        self.maxiter = maxiter
        self.a0 = a0
        self.b0 = b0

    def fit(self, X, y=None):
        # X is a list of ScalingDataset objects
        self.fss = FiniteSizeScaling(*X, s_factor=self.s_factor, k=self.k,
                                     method=self.method, maxiter=self.maxiter,
                                     a0=self.a0, b0=self.b0)
        self.fss.fit()
        return self

    def score(self, X, y=None):
        return self.fss.score(None, None)
    
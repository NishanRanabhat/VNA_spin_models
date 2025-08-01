import numpy as np 
from data_set import ScalingDataset
from dynamical_fss import FiniteSizeScaling

"""
This is a mock example to use the code
"""

#prepare data
#each dataset should have a system size L, the domain x, the range y, and the error in y
#then use these to create a ScalingDataset object

L1 = 10
x1 = np.arange(1,10,1)
y1 = np.arange(1,10,1)**2
err1 = np.arange(1,10,1)*0.001
ds1 = ScalingDataset(L1,x1,y1,err1) 

L2 = 20
x2 = np.arange(1,10,1)
y2 = np.arange(1,10,1)**2
err2 = np.arange(1,10,1)*0.001
ds2 = ScalingDataset(L2,x2,y2,err2) 

L3 = 10
x3 = np.arange(1,10,1)
y3 = np.arange(1,10,1)**2
err3 = np.arange(1,10,1)*0.001
ds3 = ScalingDataset(L3,x3,y3,err3)

#initiate a FiniteSizeScaling object

s_factor = 1.0 #spline smoothing factor multiplier (default 1.0)
k = 3 #spline degree (default 3)
method = 'Nelder-Mead' #optimizer method for fitting (default 'Nelder-Mead')
maxiter = 1000 #maximum iterations for optimizer (default 1000)
a0, b0 = 1.0, 0.5 #initial guesses for scaling exponents a and b

fss = FiniteSizeScaling(ds1,ds2,ds3,s_factor=s_factor,k=k,method=method,maxiter=maxiter,a0=a0,b0=b0)

#call fit to extract the critical exponents a and b corresponding to the best collapse
a,b = fss.fit()

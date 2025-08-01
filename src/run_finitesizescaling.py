import numpy as np
from data_set import DataSet
from finitesizescaling import FSS


#from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    """
    Prepare the dataset here. The dataset has three parts;

    system_size_list (1D array): a list of system size
    domain_list (1D array): a list of domain values
    range_list (2D array) : a 2D array range values for every system_size

    Here, we are just using a .npy files outside the directory and creating the dataset by hand. 
    """

    Spec_200 = np.load("../Central_Singular_Val_N=200_chi_max=256_a=1.80_h=0.00.npy",allow_pickle = True)[0:1000,:]
    Spec_250 = np.load("../Central_Singular_Val_N=250_chi_max=256_a=1.80_h=0.00.npy",allow_pickle = True)[0:1000,:]
    Spec_300 = np.load("../Central_Singular_Val_N=300_chi_max=256_a=1.80_h=0.00.npy",allow_pickle = True)[0:1000,:]
    Spec_350 = np.load("../Central_Singular_Val_N=350_chi_max=256_a=1.80_h=0.00.npy",allow_pickle = True)
    Spec_400 = np.load("../Central_Singular_Val_N=400_chi_max=256_a=1.80_h=0.00.npy",allow_pickle = True)

    Schmidt_gap_200 = Spec_200[:,0]**2 - Spec_200[:,1]**2
    Schmidt_gap_250 = Spec_250[:,0]**2 - Spec_250[:,1]**2
    Schmidt_gap_300 = Spec_300[:,0]**2 - Spec_300[:,1]**2
    Schmidt_gap_350 = Spec_350[:,0]**2 - Spec_350[:,1]**2
    Schmidt_gap_400 = Spec_400[:,0]**2 - Spec_400[:,1]**2

    SG_list = np.empty((5,1000))
    SG_list[0,:] = Schmidt_gap_200 
    SG_list[1,:] = Schmidt_gap_250 
    SG_list[2,:] = Schmidt_gap_300
    SG_list[3,:] = Schmidt_gap_350
    SG_list[4,:] = Schmidt_gap_400


    SG_sucep_200 = -1*np.gradient(Schmidt_gap_200,0.002)
    SG_sucep_250 = -1*np.gradient(Schmidt_gap_250,0.002)
    SG_sucep_300 = -1*np.gradient(Schmidt_gap_300,0.002)
    SG_sucep_350 = -1*np.gradient(Schmidt_gap_350,0.002)
    SG_sucep_400 = -1*np.gradient(Schmidt_gap_400,0.002)

    #range_list
    SG_sucept_list = np.empty((5,1000))
    SG_sucept_list[0,:] = SG_sucep_200 
    SG_sucept_list[1,:] = SG_sucep_250 
    SG_sucept_list[2,:] = SG_sucep_300 
    SG_sucept_list[3,:] = SG_sucep_350 
    SG_sucept_list[4,:] = SG_sucep_400 

    #system_size_list
    L_list =np.asarray([200,250,300,350,400])

    #domain_list
    beta_list = (np.arange(1,1001)*0.002)

    #call DataSet class 
    dataset = DataSet(L_list,beta_list,SG_sucept_list)

    #initial guess for critical exponents
    initial_params = (0.4,-0.32,0.38)
    
    #bounds on critical exponents
    param_bounds = [(0.39, 0.41), (-0.34, -0.32), (0.36, 0.41)]

    #scaling windows around criticality
    scaling_window = (-0.5,0.5)

    #call FSS class
    FSS = FSS(dataset,poly_order=10,initial_params=initial_params, param_bounds=param_bounds,scaling_window=scaling_window, optimization_routine="L-BFGS-B")

    #optimize the critical exponents
    val,err = FSS.optimization()

    print(val)
    print(err)

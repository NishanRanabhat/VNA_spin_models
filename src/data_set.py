import numpy as np

class DataSet:
    def __init__(self, system_size_list, domain_list, range_list):
        """
        Initialize a dataset.
        
        Args:
            system_size_list (array): The size L of the system.
            temperatures_list (1D array): Temperature or control parameter values.
            observables_list (2D array): Measured observable values.
        """
        self.system_size_list = system_size_list
        self.domain_list = domain_list
        self.range_list = range_list
        
    def validate(self):
        # Basic validation of data length consistency, etc.
        if len(self.domain_list) != self.range_list.shape[1]:
            raise ValueError("Length of domain and range must match.")
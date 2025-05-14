from neural_network_ansatz import binary_disordered_RNNwavefunction
import torch

model = binary_disordered_RNNwavefunction('vanilla', 200, 2, 1, 
                                          activation="relu",units=10, type=torch.float32, device='cpu')
print(model.n_layers)
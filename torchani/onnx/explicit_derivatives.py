# these are the explicit derivatives for ANIModel, Ensemble and EnergyShifter
# these derivatives are needed for TRT to actually be able to calculate 
# the forces in a forward pass
import torch
from torch import Tensor

def celu_backwards(input_ : Tensor, alpha : float, grad_output : Tensor):
    grad_input = grad_output.clone()
    grad_input[ input_ < 0] = torch.exp(input_ / alpha)


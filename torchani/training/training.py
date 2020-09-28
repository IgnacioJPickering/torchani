import torch
import math
from torchani.units import hartree2kcalmol

def validate_energies(model, validation):
    # get device from an arbitrary model tensor
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    for conformation in validation:
        species = conformation['species'].to(device)
        coordinates = conformation['coordinates'].to(device).float()
        true_energies = conformation['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))

class RootAtomsLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        return (self.mse(predicted, target) / num_atoms.sqrt()).mean()

def init_traditional(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_sharp_gaussian(m, scale=0.001):
    if isinstance(m, torch.nn.Linear):
        # if this is an output linear layer use a smaller scale
        # for init
        if m.weight.shape[0] == 1:
            torch.nn.init.normal_(m.weight, mean=0.0, std=scale)
        else:
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def reproducible_init_nobias(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight, 1.)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.)

def reproducible_init_bias(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight, 1.)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 1.e-5)


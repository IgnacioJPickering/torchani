import torch
import math
from torchani.units import hartree2kcalmol, hartree2ev


def validate_energies(model, validation):
    model.eval()
    # get device from an arbitrary model tensor
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for conformation in validation:
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            true_energies = conformation['energies'].to(device).float()

            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]
    main_metric = {'validation_rmse': hartree2kcalmol(math.sqrt(total_mse / count))}
    metrics = None
    model.train()
    return main_metric, metrics

def validate_energies_ex(model, validation):
    model.eval()
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    mse = torch.nn.MSELoss(reduction='none')
    total_mses = torch.zeros(model.neural_networks.num_outputs, device=device, dtype=torch.float)
    count = 0
    with torch.no_grad():
        for conformation in validation:
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            target_ground = conformation['energies'].to(device).float().reshape(-1, 1)
            target_ex = conformation['energies_ex'].to(device).float()
            target = torch.cat((target_ground, target_ex), dim=-1)

            _, predicted_energies = model((species, coordinates))
            total_mses += mse(predicted_energies, target).sum(dim=0)
            count += predicted_energies.shape[0]

    rmses_hartree = torch.sqrt(total_mses / count)
    average_rmse = hartree2kcalmol(rmses_hartree).mean()
    main_metric = {'average_rmse_kcalmol': average_rmse}
    metrics = { f'state_{j}/rmse_kcalmol': v for j, v in enumerate(hartree2kcalmol(rmses_hartree))}
    metrics = { f'state_{j}/rmse_eV': v for j, v in enumerate(hartree2ev(rmses_hartree))}
    model.train()
    return main_metric, metrics

def validate_energies_ex_and_foscs(model, validation):
    model.eval()
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    mse = torch.nn.MSELoss(reduction='none')
    total_mses = torch.zeros(model.neural_networks.num_outputs, device=device, dtype=torch.float)
    count = 0
    with torch.no_grad():
        for conformation in validation:
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            target_ground = conformation['energies'].to(device).float().reshape(-1, 1)
            target_ex = conformation['energies_ex'].to(device).float()
            target_foscs = conformation['foscs'].to(device).float()
            target = torch.cat((target_ground, target_ex, target_foscs), dim=-1)

            _, predicted_energies, predicted_foscs = model((species, coordinates))
            predicted = torch.cat((predicted_energies, predicted_foscs), dim=-1)
            total_mses += mse(predicted, target).sum(dim=0)
            count += predicted_energies.shape[0]

    num_outputs = model.neural_networks.num_outputs
    rmses_au = torch.sqrt(total_mses / count)
    average_rmse_energies = hartree2kcalmol(rmses_au[0:num_outputs]).mean()
    average_rmse_foscs = rmses_au[num_outputs:].mean()

    main_metric = {'average_rmse_energies_kcalmol': average_rmse_energies}
    metrics = { f'state_{j}/rmse_energies_kcalmol': v for j, v in enumerate(hartree2kcalmol(rmses_au[0:num_outputs]))}
    metrics.update({ f'state_{j}/rmse_energies_eV': v for j, v in enumerate(hartree2ev(rmses_au[0:num_outputs]))})
    metrics.update({ f'state_{j}/rmse_foscs_au': v for j, v in enumerate(rmses_au[num_outputs:])})
    metrics.update({ f'average_rmse_foscs_au': v for j, v in enumerate(average_rmse_foscs[num_outputs:])})
    model.train()
    return main_metric, metrics

class RootAtomsLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        return (self.mse(predicted, target) / num_atoms.sqrt()).mean()

class MultiTaskLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=10):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            self.register_buffer('weights', torch.ones(num_inputs,
                dtype=torch.double) * 1 / num_inputs)

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        squares = self.mse(predicted, target)
        losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskUncertaintyLoss(torch.nn.Module):

    def __init__(self, num_inputs=10):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        # predict log_sigmas_squared since it is more numerically stable
        # this is equivalent to initializing sigmas as ones
        self.register_parameter('log_sigmas_squared', torch.nn.Parameter(torch.zeros(num_inputs, dtype=torch.double)))

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        squares = self.mse(predicted, target) 
        losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        loss = (0.5 * torch.exp(-self.log_sigmas_squared) * losses).sum() 
        loss = loss + 0.5 * self.log_sigmas_squared.sum()
        return  loss, losses.detach()

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


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
    metrics.update({ f'state_{j}/rmse_eV': v for j, v in enumerate(hartree2ev(rmses_hartree))})
    model.train()
    return main_metric, metrics

def validate_energies_ex_and_foscs(model, validation):
    model.eval()
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    mse = torch.nn.MSELoss(reduction='none')
    total_mses = torch.zeros(model.neural_networks.num_outputs + model.neural_networks.num_other_outputs, device=device, dtype=torch.float)
    count = 0
    with torch.no_grad():
        for conformation in validation:
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            target_ground = conformation['energies'].to(device).float().reshape(-1, 1)
            target_ex = conformation['energies_ex'].to(device).float()
            target_foscs = conformation['foscs_ex'].to(device).float()
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

    metrics.update({ f'average_rmse_foscs_au': average_rmse_foscs})
    metrics.update({ f'state_{j}/rmse_foscs_au': v for j, v in enumerate(rmses_au[num_outputs:])})
    model.train()
    return main_metric, metrics

class RootAtomsLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        return (self.mse(predicted, target) / num_atoms.sqrt()).mean()

class BareLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        return (self.mse(predicted, target)).mean()

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

class MultiTaskRelativeLoss(torch.nn.Module):
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
        square_ground = self.mse(predicted[:, 0], target[:, 0])
        ratios_ex = predicted[:, 1:]/target[:, 1:]
        squares_ex = self.mse(ratios_ex, torch.ones_like(ratios_ex))
        squares = torch.cat((square_ground.unsqueeze(-1), squares_ex), dim=-1)
        losses = (squares).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskSpectraLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11, num_other_inputs=10, dipoles=False):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            if dipoles:
                # I will rescale the dipoles manually with weights if needed
                self.register_buffer('weights', torch.ones(num_inputs + 3 * num_other_inputs,
                    dtype=torch.double) * 1 / (num_inputs + 3 * num_other_inputs))
            else:
                self.register_buffer('weights', torch.ones(num_inputs + num_other_inputs,
                    dtype=torch.double) * 1 / (num_inputs + num_other_inputs))

        self.num_other_inputs = num_other_inputs
        self.dipoles = dipoles

    def forward(self, predicted, target,  target_other, predicted_other, species):

        target = torch.cat((target, target_other), dim=-1)
        predicted = torch.cat((predicted, predicted_other), dim=-1)
        losses = (self.mse(predicted, target)).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskBareLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            self.register_buffer('weights', torch.ones(num_inputs,
                dtype=torch.double) * 1 / num_inputs)

    def forward(self, predicted, target, species):
        losses = self.mse(predicted, target).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskPairwiseLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            pairs = num_inputs * (num_inputs - 1) // 2
            self.register_buffer('weights', torch.ones(pairs,
                dtype=torch.double) * 1 / pairs)

        row_major = torch.arange(0, num_inputs* num_inputs).reshape(num_inputs, num_inputs)
        idxs = torch.triu(row_major, diagonal=1)
        idxs = torch.masked_select(idxs, idxs != 0)
        self.register_buffer('idxs', idxs)

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)

        # get absolute energies
        predicted[1:] = predicted[1:] + predicted[0]
        target[1:] = target[1:] + target[0]

        # get all pairwise differences
        diff = target.unsqueeze(2) - target.unsqueeze(1)
        diff_predicted = predicted.unsqueeze(2) - predicted.unsqueeze(1)

        diff_target = diff.flatten(start_dim=1)[:, self.idxs]
        diff_predicted = diff_predicted.flatten(start_dim=1)[:, self.idxs]

        squares = self.mse(diff_predicted, diff_target)
        losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        loss = (losses * self.weights).sum()
        # I don't need one million losses so I don't even output them
        return  loss, None

class MultiTaskUncertaintyLoss(torch.nn.Module):

    def __init__(self, num_inputs=10, weight_sqrt_atoms=True):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        self.weight_sqrt_atoms = weight_sqrt_atoms
        # predict log_sigmas_squared since it is more numerically stable
        # this is equivalent to initializing sigmas as ones
        self.register_parameter('log_sigmas_squared', torch.nn.Parameter(torch.zeros(num_inputs, dtype=torch.double)))

    def forward(self, predicted, target, species):
        squares = self.mse(predicted, target) 
        if self.weight_sqrt_atoms:
            num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
            losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        else:
            losses = (squares).mean(dim=0)
        loss = (0.5 * torch.exp(-self.log_sigmas_squared) * losses).sum() 
        loss = loss + 0.5 * self.log_sigmas_squared.sum()
        return  loss, losses.detach()

def init_traditional(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_all_sharp(m, scale=0.01):
    if isinstance(m, torch.nn.Linear):
        # if this is an output linear layer use a smaller scale
        # for init
        #if m.weight.shape[0] == 1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=scale)
        #else:
        #    torch.nn.init.kaiming_normal_(m.weight, a=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_sharp_gaussian(m, scale=0.001):
    if isinstance(m, torch.nn.Linear):
        # if this is an output linear layer use a smaller scale
        # for init
        # this is a hack for now
        if m.weight.shape[0] == 1 or m.weight.shape[0] == 10:
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

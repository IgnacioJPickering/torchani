import torch
import math
import numpy as np
import pickle
from tqdm import tqdm
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

def save_histogram_energies(model, validation, ex = None):
    assert isinstance(ex, (int, None))
    model.eval()
    # get device from an arbitrary model tensor
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    with torch.no_grad():
        errors = []
        atoms = []
        true_energies_list = []
        predicted_energies_list = []
        for conformation in tqdm(validation):
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            true_energies = conformation['energies'].to(device).float()
            if ex is not None:
                true_energies_ex = conformation['energies_ex'].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
            _, predicted_energies, _ = model((species, coordinates))

            true_energies = hartree2ev(true_energies_ex)[:, ex - 1]
            predicted_energies = hartree2ev(predicted_energies[:, 1:])[:, ex - 1]

            errors.append(torch.abs(predicted_energies - true_energies).cpu().numpy())
            atoms.append(num_atoms.cpu().numpy())
            true_energies_list.append(true_energies.cpu().numpy())
            predicted_energies_list.append(predicted_energies.cpu().numpy())

        errors = np.concatenate(errors)
        atoms = np.concatenate(atoms)
        true_energies_list = np.concatenate(true_energies_list)
        predicted_energies_list = np.concatenate(predicted_energies_list)
    if ex is not None:
        suffix = f'_ex{ex}'
    else:
        suffix = ''
    with open(f'true_energies{suffix}.pkl', 'wb') as f:
        pickle.dump(true_energies_list, f)
    with open(f'predicted_energies{suffix}.pkl', 'wb') as f:
        pickle.dump(predicted_energies_list, f)
    with open(f'errors_energies{suffix}.pkl', 'wb') as f:
        pickle.dump(errors, f)
    with open(f'atoms{suffix}.pkl', 'wb') as f:
        pickle.dump(atoms, f)
    model.train()

def save_histogram_foscs(model, validation, ex=1):
    assert isinstance(ex, (int))
    model.eval()
    # get device from an arbitrary model tensor
    device = model.aev_computer.ShfR.device
    # run energy validation on a given dataset
    with torch.no_grad():
        errors = []
        atoms = []
        true_foscs_list = []
        predicted_foscs_list = []
        for conformation in tqdm(validation):
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            true_foscs = conformation['foscs_ex'].to(device).float()[:, ex-1]
            predicted_foscs = model((species, coordinates))[-1][:, ex-1]
            num_atoms = (species >= 0).sum(dim=1, dtype=true_foscs.dtype)
            errors.append(torch.abs(predicted_foscs - true_foscs).cpu().numpy())
            atoms.append(num_atoms.cpu().numpy())
            true_foscs_list.append(true_foscs.cpu().numpy())
            predicted_foscs_list.append(predicted_foscs.cpu().numpy())

        errors = np.concatenate(errors)
        atoms = np.concatenate(atoms)
        true_foscs_list = np.concatenate(true_foscs_list)
        predicted_foscs_list = np.concatenate(predicted_foscs_list)

    suffix = f'_ex{ex}'
    with open(f'true_foscs{suffix}.pkl', 'wb') as f:
        pickle.dump(true_foscs_list, f)
    with open(f'predicted_foscs{suffix}.pkl', 'wb') as f:
        pickle.dump(predicted_foscs_list, f)
    with open(f'errors_foscs{suffix}.pkl', 'wb') as f:
        pickle.dump(errors, f)
    with open(f'atoms{suffix}.pkl', 'wb') as f:
        pickle.dump(atoms, f)
    model.train()

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
    metrics.update({ f'state_{j+1}/rmse_foscs_au': v for j, v in enumerate(rmses_au[num_outputs:])})
    model.train()
    return main_metric, metrics

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


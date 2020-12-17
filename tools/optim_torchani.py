import torch
import torchani
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model2 = torchani.models.ANI2x(periodic_table_index=True).to(device).double()

#with open('/home/ignacio/Git-Repos/torchani/training_outputs/ani1x_softplus/ani1x_softplus.yaml') as f:
with open('./ani1x_gelu_rxn.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = torchani.modules.TemplateModel.from_yaml(config).to(device).shift_before_output_(True)
path = './best_rxn.pt'
model.load_state_dict(torch.load(path)['model'])
with open('./self_energies.pkl', 'rb') as f:
    model.energy_shifter.self_energies = pickle.load(f)
model = model.double().to(device)
species = torch.tensor([[1, 1, 8]], device=device)
model = torch.load('./rxn_model.pt').to(device)

eq = math.pi / 180 * 104.51
eq = 1.8240
eq_d = 0.9575
energies = []
forces_di = []
hessians_di = []
forces_ang = []
hessians_ang = []
delta = 1.0
angles = np.linspace(eq - delta, eq + delta, 1000)
def get_plots(model):
    for t in tqdm(angles):
        d = torch.tensor([0.9575], dtype=torch.double, device=device, requires_grad=True)
        theta = torch.tensor([t], dtype=torch.double, device=device, requires_grad=True)
        zero = torch.zeros(1, dtype=torch.double, device=d.device)
        upper = torch.cat((d,zero, zero))
        mid = torch.cat((d * torch.cos(theta), d * torch.sin(theta), zero))
        lower = torch.cat((zero, zero, zero))
        coords = torch.stack((upper, mid, lower), dim=0).unsqueeze(0)
    
        energy = model((species, coords)).energies
        force_ang = - torch.autograd.grad(energy.sum(), theta, create_graph=True)[0]
        hess_ang = torch.autograd.grad(force_ang, theta, retain_graph=True)[0]
    
        energy = model((species, coords)).energies
        forces_ang.append(force_ang.item())
        hessians_ang.append(hess_ang.item())
        energies.append(energy.item())
    return energies, forces_ang, hessians_ang

fig, ax = plt.subplots(2, 3, sharex=True)

energies, forces_ang, hessians_ang = get_plots(model2)
ax[0, 0].scatter(angles, energies, label='energy', s=0.8)
ax[0, 1].scatter(angles, forces_ang, label='force', s=0.8)
ax[0, 2].scatter(angles, hessians_ang, label='second', s=0.8)
ax[0, 0].plot(angles, energies, linewidth=0.5)
ax[0, 1].plot(angles, forces_ang, linewidth=0.5)
ax[0, 2].plot(angles, hessians_ang, linewidth=0.5)
ax[0, 0].set_xlim(eq-delta, eq+delta)
ax[0,0].set_title('celu, E')
ax[0,1].set_title('celu, F')
ax[0,2].set_title('celu, H')
with open('plot_force.csv', 'w') as f:
    f.write('#angles (rad), forces (hartree/angstrom)\n')
    for a, fo in zip(angles, forces_ang):
        f.write(f'{a} {fo}\n')
energies = []
forces_di = []
hessians_di = []
forces_ang = []
hessians_ang = []
energies, forces_ang, hessians_ang = get_plots(model)
ax[1, 0].scatter(angles, energies, label='energy', s=0.8)
ax[1, 1].scatter(angles, forces_ang, label='force', s=0.8)
ax[1, 2].scatter(angles, hessians_ang, label='second', s=0.8)
ax[1, 0].plot(angles, energies, linewidth=0.5)
ax[1, 1].plot(angles, forces_ang, linewidth=0.5)
ax[1, 2].plot(angles, hessians_ang, linewidth=0.5)
ax[1, 0].set_xlim(eq-delta, eq+delta)
ax[1,0].set_title('gelu, E')
ax[1,1].set_title('gelu, F')
ax[1,2].set_title('gelu, H')
plt.legend()
plt.show()


plt.show()

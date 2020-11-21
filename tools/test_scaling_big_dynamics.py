import torch
import torchani
from torchani.modules import AEVComputerNL, AEVComputerJoint
from torchani.geometry import tile_into_cube
from time import time
import pickle
double = False
device = torch.device('cuda')
model = torchani.models.ANI1x(model_index=0).to(device)
aevc = AEVComputerNL.like_ani1x().to(device)
#aevc = AEVComputerJoint.like_ani1x().to(device)
model.aev_computer = aevc
string = 'dp_times_sizes_old'
if double:
    model = model.double()

from tqdm import tqdm
times = []
sizes = []
# 59 for the other
for r in tqdm(range(1, 60)):
    cut = 5.2
    species = torch.zeros(4).unsqueeze(0).to(torch.long).to(device)
    coordinates = torch.randn(4, 3).unsqueeze(0).to(device)*(cut - 0.001)
    species, coordinates = tile_into_cube((species, coordinates), box_length=5.1, repeats=r)
    cell_size = cut*r + 0.1
    eps = 1e-5 
    coordinates = torch.clamp(coordinates, min=eps, max=cell_size - eps)
    cell = torch.diag(torch.tensor([cell_size, cell_size, cell_size])).float().to(device)

    pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
    coordinates.requires_grad_(True)
    start = time()
    if double:
        coordinates = coordinates.double()
        cell = cell.double()
    _, energy = model((species, coordinates), cell=cell, pbc=pbc)
    force = - torch.autograd.grad(energy.sum(), coordinates)[0]
    stop = time()
    times.append(stop - start)
    sizes.append(species.shape[1])
    del species
    del energy
    del force
    del coordinates
    del cell
    #print(r, species.shape, energy, stop - start)

with open(string + '.pkl', 'wb') as f:
    pickle.dump({'times':times, 'sizes': sizes}, f)

import torch
import numpy as np
import torchani
from torchani import geometry
import ase
from ase import units
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from tqdm import tqdm
import math
import os
#path = '/media/samsung1TBssd/example_liquid.traj'
#traj = Trajectory(path)
#for j, step in tqdm(enumerate(traj)):
#    tmp = f'/media/samsung1TBssd/tmp{j}.xyz'
#    write(tmp, step)
#    with open(tmp, 'r') as f:
#        fstring = f.read()
#    os.unlink(tmp)
#    with open('/media/samsung1TBssd/example_liquid.xyz', 'a') as f:
#        f.write(fstring)
#os.unlink(path)
#exit()

def make_methane(device=None, eq_bond = 1.09):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = eq_bond * 2 / math.sqrt(3)
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0.5, 0.5, 0.5]]],
        device=device, dtype=torch.double) * d
    species = torch.tensor([[1, 1, 1, 1, 6]], device=device, dtype=torch.long)
    return species, coordinates.double()

def make_water(device=None, eq_bond = 0.957582):
    d = eq_bond
    t = math.pi / 180 * 104.485
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coordinates = torch.tensor([[d, 0, 0],
                                [d * math.cos(t), d * math.sin(t), 0], 
                                [0, 0, 0]], device=device).double()
    species = torch.tensor([[1, 1, 8]], device=device, dtype=torch.long)
    return species, coordinates.double()

def tensors_from_xyz(file_path, device=None, convert=False):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coordinates = np.loadtxt(file_path, skiprows=2, usecols=[1, 2, 3])
    species = np.loadtxt(file_path, skiprows=2, usecols=[0], dtype=str).tolist()
    species = np.asarray([torchani.utils.PERIODIC_TABLE.index(s) for s in species], dtype=int)
    species = torch.from_numpy(species).to(dtype=torch.long, device=device).unsqueeze(0)
    coordinates = torch.from_numpy(coordinates).to(dtype=torch.double, device = device).unsqueeze(0)
    if convert:
        converter = torchani.nn.SpeciesConverter(['H', 'C', 'N', 'O'])
        species, _ = converter((species, coordinates))
    with open(file_path, 'r') as f:
        cell_string = f.read().split('\n')[1].split('=')[-1].split()
        cell_string = torch.tensor([float(s) for s in cell_string], device = device, dtype = torch.double)
        cell = torch.diag(cell_string)
    return species, coordinates, cell

device = torch.device('cuda')
model = torchani.models.ANI1x(periodic_table_index=True, cell_list=True).to(device)
#model = torchani.models.ANI1x(periodic_table_index=True).to(device)
total_steps = 1000
repeats = (5, 5, 30)
#species, coordinates = make_water(device)
#tiled_species, tiled_coord, cell = geometry.tile_into_tight_cell((species, coordinates),
#                                            noise=0.1,
#                                            repeats=repeats, density=0.0923)
tiled_species, tiled_coord, cell = tensors_from_xyz('./large_water.xyz')
molecule = ase.Atoms(tiled_species.squeeze().tolist(),
                     positions=tiled_coord.squeeze().tolist(),
                     calculator=model.ase(), pbc=True,
                     cell=cell.cpu().numpy())
write(images=molecule, filename='initial_coordinates.xyz')
tiled_coord.requires_grad_(True)
MaxwellBoltzmannDistribution(molecule, 300 * units.kB)
dyn = Langevin(molecule, 1.0 * units.fs, 300 * units.kB, 0.002)
path = './large.traj'
traj = Trajectory(path, 'w', molecule)
dyn.attach(traj.write, interval=10)

for _ in tqdm(range(total_steps//10)):
    dyn.run(10)
traj.close()

traj = Trajectory(path)
for j, step in tqdm(enumerate(traj)):
    tmp = f'./tmp{j}.xyz'
    write(tmp, step)
    with open(tmp, 'r') as f:
        fstring = f.read()
    os.unlink(tmp)
    with open('./large.xyz', 'a') as f:
        f.write(fstring)
os.unlink(path)

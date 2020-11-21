import torch
import timeit
import torchani
from torchani.modules import AEVComputerNL, AEVComputerJoint
from torchani.geometry import tile_into_cube, tile_into_tight_cell # noqa
from time import time
import copy
import math
import pickle # noqa
double = True
device = torch.device('cuda')
model = torchani.models.ANI1x(model_index=0, periodic_table_index=True).to(device)
aevc = AEVComputerNL.like_ani1x().to(device)
aevj = AEVComputerJoint.like_ani1x().to(device)
model.aev_computer = aevc
string = 'dp_times_sizes_old'
if double:
    model = model.double()

timers = dict()
def time_func(key, func):
    timers[key] = 0

    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        timers[key] += end - start
        return ret
    return wrapper

def time_functions_in_object(object_, function_names_list):
    # Wrap all the functions from "function_names_list" from the module
    # "module" with a timer
    for n in function_names_list:
        setattr(object_, n, time_func(f'{n}', getattr(object_, n)))

functions = ['neighbor_pairs']
functions_to_time = ['cutoff_cosine', 'radial_terms', 'angular_terms',
                     'compute_shifts', 'neighbor_pairs',
                     'neighbor_pairs_nopbc', 'cumsum_from_zero',
                     'triple_by_molecule', 'compute_aev']
time_functions_in_object(model.aev_computer, functions_to_time)

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

from tqdm import tqdm
times = []
sizes = []
# 59 for the other
times_functions = []

for r in tqdm(range(2, 34)):
    cut = 5.2
    species, coordinates = make_water(device)
    #species = torch.zeros(4).unsqueeze(0).to(torch.long).to(device)
    #coordinates = torch.randn(4, 3).unsqueeze(0).to(device)*(cut - 0.001)
    #species, coordinates = tile_into_cube((species, coordinates), box_length=5.1, repeats=r)
    species, coordinates, cell = tile_into_tight_cell((species, coordinates),
                                                noise=0.1,
                                                repeats=r, density=0.1)
    #cell_size = cut*r + 0.1
    #eps = 1e-5 
    #coordinates = torch.clamp(coordinates, min=eps, max=cell_size - eps)
    #cell = torch.diag(torch.tensor([cell_size, cell_size, cell_size])).float().to(device)

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
    times_functions.append(copy.deepcopy(timers))
    # reset timers
    for k, v in timers.items():
        timers[k] = 0.0

with open(string + '.pkl', 'wb') as f:
    pickle.dump({'times':times, 'sizes': sizes, 'times_functions': times_functions}, f)

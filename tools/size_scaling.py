import torch
import torchani
from torchani import geometry
import timeit
import argparse
import copy

def time_func(key, func):
    timers[key] = 0

    def wrapper(*args, **kwargs):
        if synchronize:
            torch.cuda.synchronize()
        start = timeit.default_timer()
        ret = func(*args, **kwargs)
        if synchronize:
            torch.cuda.synchronize()
        end = timeit.default_timer()
        timers[key] += end - start
        return ret

    return wrapper

def zero_timers(timers):
    timers = {k: 0 for k in timers.keys()}


def time_functions_in_module(module, function_names_list):
    # Wrap all the functions from "function_names_list" from the module
    # "module" with a timer
    for n in function_names_list:
        setattr(module, n, time_func(f'{module.__name__}.{n}', getattr(module, n)))


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-y', '--synchronize',
                        action='store_true',
                        help='whether to insert torch.cuda.synchronize() at the start and end of each function')
    parser = parser.parse_args()
    device = parser.device


    if parser.synchronize:
        synchronize = True
    else:
        synchronize = False
        print('WARNING: Synchronization creates some small overhead but if CUDA'
              ' streams are not synchronized the timings before and after a'
              ' function do not reflect the actual calculation load that'
              ' function is performing. Only run this benchmark without'
              ' synchronization if you know very well what you are doing')

    model = torchani.models.ANI1x(periodic_table_index=True, model_index=0).to(device)

    timers = {}
    # enable timers
    functions_to_time = ['cutoff_cosine', 'radial_terms', 'angular_terms',
                         'compute_shifts', 'neighbor_pairs',
                         'neighbor_pairs_nopbc', 'cumsum_from_zero',
                         'triple_by_molecule', 'compute_aev']

    time_functions_in_module(torchani.aev, functions_to_time)
    model.aev_computer.forward = time_func('total_aev', model.aev_computer.forward)
    model.neural_networks.forward = time_func('total_nn', model.neural_networks.forward)
    model.energy_shifter.forward = time_func('total_eshifter', model.energy_shifter.forward)
    model.forward = time_func('total', model.forward)
    max_repeats = 18
    times_to_average_over = 1
    timers_list = []
    atoms = []
    for r in range(max_repeats):
        zero_timers(timers)
        for _ in range(times_to_average_over):
            d = 1.0
            coordinates = torch.tensor([[[0.0, 0.0, 0.0],
                                         [-d, -d, d],
                                         [-d, d, -d],
                                         [d, -d, -d]]], device=device)
            species = torch.tensor([[6, 1, 1, 1]], device=device)
            tiled_species, tiled_coord = geometry.tile_into_cube((species, coordinates), noise=0.2, repeats=r+1)
            atoms.append(tiled_species.shape[1])
            tiled_coord.requires_grad_()
            energies = model((tiled_species, tiled_coord)).energies
            torch.autograd.grad(energies.sum(), tiled_coord)
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()

        timers_list.append(copy.deepcopy(timers))
    total_timers = {k: [] for k in timers.keys()}
    for t in timers_list:
        for k in total_timers.keys():
            total_timers[k].append(t[k])
    
    # plotting code
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for k, v in total_timers.items():
        ax.scatter(atoms, v, label=k)
        ax.plot(atoms, v, linestyle='dashed')
    plt.legend()
    plt.show()

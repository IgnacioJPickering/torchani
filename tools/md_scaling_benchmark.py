import torch
import torchani
import time
import pickle
import ase
import numpy as np
from tqdm import tqdm

from torchani import geometry
from ase.md.langevin import Langevin
from ase import units

import matplotlib.pyplot as plt
from pathlib import Path

def plot_file(file_path, comment):
    with open(Path(file_path).resolve(), 'rb') as f:
        times_sizes = pickle.load(f)
        all_trials = np.asarray(times_sizes['times'])
        sizes = times_sizes['atoms']
    
    fig, ax = plt.subplots()
    std = all_trials.std(axis=0)
    mean = all_trials.mean(axis=0)
    assert len(std) == len(mean)
    assert len(std) == len(sizes)
    for times in all_trials:
        ax.errorbar(x=sizes, y=mean, yerr=std*2, ecolor='k', capsize =2, fmt='s--', ms=4)
    #ax.plot(times, sizes)
    ax.set_xlabel('System size (atoms)')
    ax.set_ylabel('Walltime per ns (h)')
    ax.set_title(comment)
    plt.show()

if __name__ == "__main__":
    import argparse
    # parse command line arguments
    parser = argparse.ArgumentParser(description='MD scaling benchmark for torchani')
    parser.add_argument('-d', '--device', type=str, default='cuda' )
    parser.add_argument('-s', '--steps', type=int, default=100)
    parser.add_argument('-b', '--box-repeats', type=int, default=15)
    parser.add_argument('-o', '--only-plot', action='store_true', default=False)
    parser.add_argument('-f', '--file-name', default=None)
    parser.add_argument('-m', '--model', default='ani1x_one')
    parser.add_argument('-t', '--trials', default=5)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    args = parser.parse_args()

    file_name = args.file_name or args.model
    pickle_file = f'{file_name}.pkl'
    csv_file = f'{file_name}.csv'
    comment = ' '.join([s.capitalize() for s in args.model.split('_')] + ['Model'])
    
    if not args.only_plot:
        num_atoms = 4
        device = torch.device(args.device)
        print(f'Running on {device} {torch.cuda.get_device_name()}')
        print(f'CUDA is avaliable: {torch.cuda.is_available()}')
        print(f'Running benchmark for {args.steps} steps')
        sizes = (num_atoms * torch.arange(1,
            args.box_repeats + 1)**3).numpy().tolist()
        print(f'Running on the following sizes: {sizes}')
        
        if args.model == 'ani1x_one':
            model = torchani.models.ANI1x(periodic_table_index=True, model_index=0).to(device)
        elif args.model == 'ani1x_all':
            model = torchani.models.ANI1x(periodic_table_index=True).to(device)
        elif args.model == 'ani2x_one':
            model = torchani.models.ANI2x(periodic_table_index=True, model_index=0).to(device)
        elif args.model == 'ani2x_all':
            model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        elif args.model == 'ani1ccx_one':
            model = torchani.models.ANI1ccx(periodic_table_index=True, model_index=0).to(device)
        elif args.model == 'ani1ccx_all':
            model = torchani.models.ANI1ccx(periodic_table_index=True).to(device)
        all_trials = [] 
        for j in range(args.trials):
            times = []
            for r in tqdm(range(args.box_repeats)):
                d = 0.5
                coordinates = torch.tensor(
                    [[[0.0, 0.0, 0.0], [-d, -d, d], [-d, d, -d], [d, -d, -d]]],
                    device=device)
                species = torch.tensor([[6, 1, 1, 1]], device=device)
                tiled_species, tiled_coord = geometry.tile_into_cube((species, coordinates),
                                                            noise=0.2,
                                                            repeats=r + 1)
                molecule = ase.Atoms(tiled_species.squeeze().tolist(),
                                     positions=tiled_coord.squeeze().tolist(),
                                     calculator=model.ase())
                tiled_coord.requires_grad_()
                dyn = Langevin(molecule, 1.0 * units.fs, 300 * units.kB, 0.2)

                start = time.time()
                dyn.run(args.steps)
                end = time.time()
                times.append(((end - start)/3600)  * 1/(args.steps * 1e-6))
            all_trials.append(times)

        with open(pickle_file, 'wb') as f:
            pickle.dump({'times':all_trials, 'atoms':sizes}, f)

        with open(csv_file, 'w') as f:
            f.write(f'#{comment}')
            titles = '#' + ' '.join([f'Trial {j} walltime per ns (h)' for j in range(args.trials)])
            titles += '\n'
            f.write(titles)
            all_trials = np.asarray(all_trials)
            for times, s in zip(all_trials, sizes):
                    string = ' '.join(times.astype(str)) + f' {s}\n'
                    f.write(string)
    if args.plot:
        plot_file(pickle_file, comment)

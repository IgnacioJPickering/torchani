import torchani
import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.TemplateModel.like_ani1x().to(device)
aev_computer = model.aev_computer
aev_computer_norm = torchani.modules.AEVComputerNorm.like_ani1x().to(device)
species_converter = model.species_converter
data_path = '/home/ignacio/Datasets/ani1x_release.pkl'
data_path = Path(data_path).resolve()

with open(data_path, 'rb') as f:
    pickled_dataset = pickle.load(f)
    training = pickled_dataset['training']
    validation = pickled_dataset['validation']

def get_histogram_for_element(j, norm=False):
    if norm:
        file_name = f'/media/samsung1TBssd/aev_plots_norm/element_{j}.pkl'
    else:
        file_name = f'/media/samsung1TBssd/aev_plots/element_{j}.pkl'
    if Path(file_name).resolve().is_file():
        print(f'Histogram for element {j} already calculated, unpickling')
        with open(file_name, 'rb') as f:
            dim_plot = pickle.load(f)
        return dim_plot['other'], dim_plot['zeros']
    
    # else calculate it
    dim_plot = []
    with torch.no_grad():
        print(f'Calculating histogram for element {j}')
        for c in tqdm(training):
            species = c['species'].to(device)
            coordinates = c['coordinates'].to(device)
            species_coordinates = (species, coordinates)
            species_coordinates = species_converter(species_coordinates)
            if norm:
                species, aevs = aev_computer_norm(species_coordinates)
            else:
                species, aevs = aev_computer(species_coordinates)
            aevs = aevs.reshape(-1, 384)
            dim_plot.append(aevs[:, j].cpu())
    dim_plot = torch.cat(dim_plot, dim=0).squeeze()
    dim_plot = dim_plot.cpu().numpy()

    nonzeros = np.count_nonzero(dim_plot)
    total = len(dim_plot)
    zeros = total - nonzeros
    nonzero_dim_plot = dim_plot[dim_plot.nonzero()]
    sparse_dict = {'zeros': zeros, 'other': nonzero_dim_plot}

    with open(file_name, 'wb') as f:
        pickle.dump(sparse_dict, f)

    return nonzero_dim_plot, zeros

# each of these has 16 elements
elements = ['H', 'C', 'N', 'O']
pairs = list(combinations_with_replacement(elements, 2))

radial_idxs = {e : s for e, s in zip(elements, np.split(np.arange(0, 64), len(elements)))}
angular_idxs = {p : s for p, s in zip(pairs, np.split(np.arange(64, 384), len(pairs)))}
for v in angular_idxs.values():
    assert len(v) == 32

for v in radial_idxs.values():
    assert len(v) == 16

threshold = 1e-2
for k, idxs in angular_idxs.items():
    fig, ax = plt.subplots(4, 8)
    fig.suptitle(f'Element Pair {k} Angular AEV')
    ax = ax.ravel()
    bins = 100
    for (j, a), i in zip(enumerate(ax), idxs):
        nonzero_dim_plot, zeros = get_histogram_for_element(i, norm=False)
        assert len(nonzero_dim_plot.shape) == 1
        nonzeros = len(nonzero_dim_plot)
        total = nonzeros + zeros
        a.hist(nonzero_dim_plot[nonzero_dim_plot > threshold], bins=bins, label=f'Feature {j}, {zeros * 100/total:.1f} % zeros')
        a.legend()
    plt.show()

for k, idxs in radial_idxs.items():
    fig, ax = plt.subplots(4, 4)
    fig.suptitle(f'Chemical Element {k} Radial AEV')
    ax = ax.ravel()
    bins = 100
    for (j, a), i in zip(enumerate(ax), idxs):
        nonzero_dim_plot, zeros = get_histogram_for_element(i, norm=False)
        assert len(nonzero_dim_plot.shape) == 1
        nonzeros = len(nonzero_dim_plot)
        total = nonzeros + zeros
        a.hist(nonzero_dim_plot, bins=bins, label=f'Feature {j}, {zeros * 100/total:.1f} % zeros')
        a.legend()
    plt.show()
exit()

bins = 100
for j in range(0, 384):
    nonzero_dim_plot, zeros = get_histogram_for_element(j)
    assert len(nonzero_dim_plot.shape) == 1
    nonzeros = len(nonzero_dim_plot)
    total = nonzeros + zeros

    print('Total', total, 'Zeros', zeros, 'Nonzeros', nonzeros )
    print('Zeros %', zeros * 100/total, 'Nonzeros', nonzeros * 100/total )

    fig, ax = plt.subplots()
    ax.hist(nonzero_dim_plot, bins=bins)
    ax.set_title(f'element {j} nonzero histogram')
    plt.show()


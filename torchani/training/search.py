"""functions useful for hyperparameter search"""
from pathlib import Path
import numpy as np
import yaml
from math import log10, floor
from scipy.stats import loguniform as scipy_loguniform
from scipy.stats import uniform as scipy_uniform
from scipy.stats import randint as scipy_randint

@np.vectorize
def round_to_sigfigs(x, n=2):
    r"""round to a given number of significant figures"""
    digits = abs(x)
    digits = floor(log10(digits))
    digits = n - int(digits) - 1
    return round(x, digits)

def log_uniform(low: float, high: float, size: int, sigfigs: int = 2):
    r"""get a log uniform ndarray with a given number of sigfigs"""
    x = scipy_loguniform(a=low, b=high).rvs(size)
    return round_to_sigfigs(x, sigfigs)

def linear_uniform(low: float, high: float, size: int, sigfigs: int = 2):
    r"""get a uniform ndarray with a given number of sigfigs"""
    x = scipy_uniform(low, high).rvs(size)
    return round_to_sigfigs(x, sigfigs)

def integer_uniform(low: float, high: float, size: int):
    r"""get a uniform ndarray of integers, high inclusive"""
    high = high + 1
    x = scipy_randint(low, high).rvs(size)
    return x

def pow2_uniform(low: float, high: float, size: int):
    r"""get a uniform ndarray of integers, high inclusive

    Note that high and low must be powers of 2!"""
    low = int(np.log2(low))
    high = high * 2
    high = int(np.log2(high))

    x = scipy_randint(low, high).rvs(size)
    return np.power(2, x).astype(int)

def insert_in_key(dict_, key, value):
    # this will find the fist key in a nested dictionary
    # that matches the wanted key and change the value there
    for k, v in dict_.items():
        # skip optimizer_loss for now, except if the key is 'loss_lr'
        if k == 'optimizer_loss' and key != 'loss_lr': continue
        if k == 'optimizer_loss' and key == 'loss_lr':
            key = 'lr'
        if isinstance(v, dict):
            key_found = insert_in_key(v, key, value)
            if key_found: 
                return True
        if k == key:
            dict_[key] = value
            return True
    return False

def move_settings_to_index(dict_, new_settings_dict, idx):
    # moves all settings in dict_ to the settings
    # specified in new_settings_dict, in the index idx
    length = 0
    for k, settings in new_settings_dict.items():
        assert isinstance(settings, dict)
        for s, v in settings.items():
            assert isinstance(v, list), "All values in settings should be lists"
            if length == 0: length = len(v)
            assert len(v) == length, "All values in settings should have the same length"
            try:
                dict_[k][s] = v[idx]
            except IndexError as e:
                print(f"Attempted scan search with index {idx} but it is already complete")
                raise e
        print(f'Using {v[idx]} for [{k}][{s}] in scan search')
    return dict_

# the hyperparameter search setup is saved in a dictionary 
# with bounds, if the search is linear or logarithmic, integer or a list of
# ints

# dims are the dimensions of the atomic networks, here the trials and range are a bit different
# trials is a 1D array, the len of trials will be the number of layers
# ranges is a 2D array, if a range is the same for lower and upper then that 
# dimension is fixed. 


# trials should be the same for all the things I'm measuring in hyperparameter search, 
# if there is something I don't want to change then I can just fix it and be 
# done with it
trials = 3
setup_ranges = {'weight_decay' : [1e-5, 1e-7], 
                'lr': [1e-7, 1e-3], 
                'zeta': [15., 32.], 
                'dims': [[128, 200], [130, 115], [50, 96]]}


def get_one_random_instance(parameter, range_):
    return get_random_parameter_scan(parameter, range_, 1)[0]


def get_random_parameter_scan(parameter, range_, trials):
    # given a parameter name, and some range and number of trials specification
    # this function returns an array which has trials for that parameter
    log_uniform_params = ['weight_decay', 'lr']
    integer_params = ['angle_sections', 'radial_dist_divisions', 'angular_dist_divisions']
    pow2_params = ['batch_size']
    linear_uniform_params = ['zeta', 'angular_eta', 'radial_eta', 'radial_start',
            'angular_start', 'radial_cutoff', 'angular_cutoff']
    
    if parameter in log_uniform_params:
        return log_uniform(low=range_[0], high=range_[1], size=trials).tolist()
    elif parameter in pow2_params:
        return pow2_uniform(low=range_[0], high=range_[1], size=trials).tolist()
    elif parameter in linear_uniform_params:
        return linear_uniform(low=range_[0], high=range_[1], size=trials).tolist()
    elif parameter in integer_params:
        return integer_uniform(low=range_[0], high=range_[1], size=trials).tolist()
    elif parameter == 'dims':
        range_ = np.asarray(range_)
        output = [integer_uniform(r[0], r[1], trials) for r in range_]
        output = np.asarray(output).transpose().tolist()
        return output

def generate_trials(setup_ranges, trials, model_path, verbose=False):
    if verbose:
        print(f'Generating {trials} training setups for random hyperparameter search,'
                f' scanning parameters {set(setup_ranges.keys())}')
    if isinstance(model_path, str):
        model_path = Path(model_path).resolve()
    else:
        model_path = model_path.resolve()
    copies = []
    for _ in range(trials):
        with open(model_path, 'r') as f:
            copies.append(yaml.load(f, Loader=yaml.FullLoader))
    
    for p in setup_ranges.keys():
        setup_ranges[p] = get_random_parameter_scan(p, setup_ranges[p], trials)
    
    for j in range(trials):
        for k, v in setup_ranges.items():
            insert_in_key(copies[j], k, v[j])
    
    for j, c in enumerate(copies):
        aev = c['aev_computer']
        radial_length = aev['num_species'] * aev['radial_dist_divisions']
        pairs = (aev['num_species'] * (aev['num_species'] + 1)) // 2
        angular_length = pairs * aev['angular_dist_divisions'] * aev['angle_sections']
        total_length = angular_length + radial_length
        
        # These always have to be changed for consistency between the modules 
        c['atomic_network']['dim_in'] = total_length
        num_species = len(c['species_converter']['species'])
        assert c['aev_computer']['num_species'] == num_species
        assert c['energy_shifter']['num_species'] == num_species
    return copies

import numpy as np
import scipy
import matplotlib.pyplot as plt
def analyze_lr_scan(path, log=True, max_epoch=25):
    # input is the path with the location of the lr scan
    if isinstance(path, str):
        path = Path(path).resolve()
    assert path.is_dir()
    lrs = []
    slopes = []
    rmses = []
    epochs = []
    intercepts = []
    for trial_path in path.iterdir():
        if trial_path.is_dir():
            for f in trial_path.iterdir():
                if f.suffix == '.yaml':
                    with open(f, 'r') as file_:
                        config = yaml.load(file_, Loader=yaml.FullLoader)
                    lrs.append(config['optimizer']['lr'])
                if f.suffix == '.csv':
                    with open(f, 'r') as file_:
                         rmse, epoch, _ = np.loadtxt(file_, unpack=True, skiprows=1, usecols=(0, 1, 2))
                    if log:
                        rmse = np.log(rmse)
                    try:
                        rmse = rmse[:max_epoch]
                        epoch = epoch[:max_epoch]
                        assert len(epoch) == max_epoch
                    except AssertionError:
                        print(f'Warning, {f} has only {len(epoch)} epochs')
                    m, b, r, _, _ =  scipy.stats.linregress(epoch, rmse)
                    slopes.append(m)
                    rmses.append(rmse)
                    epochs.append(epoch)
                    intercepts.append(b)
    if log:
        ylabel1 = r'$\partial$ ln (RMSE / 1 kcal/mol) / $\partial$ epoch'
        ylabel2 = r'ln (RMSE / 1 kcal/mol)'
    else:
        ylabel1 = r'$\partial$ RMSE / $\partial$ epoch (kcal/mol)'
        ylabel2 = r'RMSE (kcal/mol)'
    fig, ax = plt.subplots(1, 2)
    ax[0].set_xscale('log')
    ax[0].scatter(lrs, slopes)
    ax[0].set_xlabel(r'Learning rate, $\lambda$')
    ax[0].set_ylabel(ylabel1)

    colors = [plt.cm.jet(j) for j in np.linspace(0, 1, len(lrs))]
    for lr, rmse, epoch, slope, intercept, c in zip(lrs, rmses, epochs, slopes, intercepts, colors):
        ax[1].plot(epoch, rmse, label=f'lr = {lr}',color=c)
        ax[1].plot(epoch, epoch * slope + intercept, color=c)
        ax[1].set_xlabel(r'epoch number')
        ax[1].set_ylabel(ylabel2)
    ax[1].legend()
    plt.show()
        
def dump_to_files(copies, parent_dir = '.'):
    name = copies[0]['name']
    for c in copies:
        assert name == c['name']
    parent_dir = Path(parent_dir).resolve()
    main_dir = parent_dir.joinpath(name)
    # dump yaml dictionaries to files, maintain order to 
    # improve readability
    # each of the copies is stored in its own directory together with the
    # best.pt and latest.pt models and the tb events
    for j, c in enumerate(copies):
        file_path = main_dir.joinpath(f'trial_{j}/trial_{j}.yaml')
        file_path.parent.mkdir(parents=True)
        with open(file_path, 'w') as f:
            # needs a very new version of yaml
            yaml.dump(c, f, sort_keys=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to analyze learning rate scan')
    parser.add_argument('-m', '--max-epoch', default=20, type=int, help='Maximum epoch to take into account for linear regression')
    args = parser.parse_args()
    path = args.path
    max_epoch = args.max_epoch
    analyze_lr_scan(Path(path).resolve(), max_epoch=max_epoch)


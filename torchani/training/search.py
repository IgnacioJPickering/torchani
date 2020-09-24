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
        if isinstance(v, dict):
            key_found = insert_in_key(v, key, value)
            if key_found: 
                return True
        if k == key:
            dict_[key] = value
            return True
    return False

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
        with open('./hyper.yaml', 'r') as f:
            copies.append(yaml.load(f, Loader=yaml.FullLoader))
    
    for p in setup_ranges.keys():
        setup_ranges[p] = get_random_parameter_scan(p, setup_ranges[p], trials)
    
    for j in range(trials):
        for k, v in setup_ranges.items():
            insert_in_key(copies[j], k, v[j])
    
    for j, c in enumerate(copies):
        aev = c['aev_computer']['kwargs']
        radial_length = aev['num_species'] * aev['radial_dist_divisions']
        pairs = (aev['num_species'] * (aev['num_species'] + 1)) // 2
        angular_length = pairs * aev['angular_dist_divisions'] * aev['angle_sections']
        total_length = angular_length + radial_length
        
        # These always have to be changed for consistency between the modules 
        c['atomic_network']['kwargs']['dim_in'] = total_length
        num_species = len(c['species_converter']['kwargs']['species'])
        assert c['aev_computer']['kwargs']['num_species'] == num_species
        assert c['energy_shifter']['kwargs']['num_species'] == num_species
    return copies
    
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

setup_ranges = {'weight_decay' : [1e-7, 1e-5], 
                'lr': [1e-7, 1e-3], 
                'dims': [[128, 170], [113, 125], [123, 150]]}
trials = 4

copies = generate_trials(setup_ranges, trials, './hyper.yaml')
dump_to_files(copies)


"""functions useful for hyperparameter search"""
#import torch
import numpy as np
import yaml
from math import log10, floor
from scipy.stats import loguniform as scipy_loguniform
from scipy.stats import uniform as scipy_uniform

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

def integer_uniform(low: float, high: float, size: int, sigfigs: int = 2):
    r"""get a uniform ndarray with a given number of sigfigs"""
    x = scipy_uniform(low, high).rvs(size)
    return round_to_sigfigs(x, sigfigs)

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

hyperparameters = ['weight_decay', 'lr']
sizes = np.asarray([3, 3, 2])
assert len(sizes) == len(hyperparameters)
num_copies = sizes.prod()
print('Generating {num_copies} training setups for hyperparameter search')

copies = []
for _ in range(num_copies):
    with open('./hyper.yaml', 'r') as f:
        copies.append(yaml.load(f, Loader=yaml.FullLoader))

for p in hyperparameters:
    key_found = insert_in_key(copies[0], p, None)
    print(copies[0])
        


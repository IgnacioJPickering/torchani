from pathlib import Path
from collections import namedtuple
from sys import maxsize
import pickle
import time
import copy

import torch
import yaml
import torch.utils.tensorboard
from tqdm import tqdm

import torchani
from torchani import data
from torchani.modules import TemplateModel
from torchani.training import validate_energies
from torchani.training import get_one_random_instance, insert_in_key

from torch import optim
from torch.optim import lr_scheduler

def load_from_checkpoint(latest_checkpoint, model, optimizer, lr_scheduler):
    if latest_checkpoint.is_file():
        checkpoint = torch.load(latest_checkpoint.as_posix())
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

def save_checkpoint(path, model, optimizer, lr_scheduler):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, path.as_posix())

def log_per_epoch(tensorboard, main_metric, lr_scheduler, optimizer, metrics=None, epoch_time=None, verbose=True, csv_file=None):
    # Other metrics is a dictionary with different metrics
    main_key, main_value = main_metric.items()
    if epoch_time is None:
        epoch_time = 0.0
    epoch_number = lr_scheduler.last_epoch
    if verbose:
        print(f': {main_key} {main_value} after epoch {lr_scheduler.last_epoch} time: {epoch_time if epoch_time is not None else 0}')
    if tensorboard is not None:
        tensorboard.add_scalar(f'{main_key}', main_value, epoch_number)
        tensorboard.add_scalar(f'best_{main_key}', lr_scheduler.best, epoch_number)
        tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'] , epoch_number)
        if metrics is not None:
            for k, v in metrics.items():
                tensorboard.add_scalar(f'{k}', v, epoch_number)
        if epoch_time is not None:
            tensorboard.add_scalar('epoch_time', epoch_time, epoch_number)
    if csv_file is not None:
        if not csv_file.is_file():
            with open(csv_file, 'w') as f:
                s = f'{main_key} epoch time lr'
                if metrics is not None:
                    for k in metrics.keys:
                        s += f' {k}'
                s += '\n'
                f.write(s)

        with open(csv_file, 'a') as f:
            s = f'{main_value} {epoch_number} {epoch_time}'
            if metrics is not None:
                for v in metrics.values():
                    s += f' {v}'
            s += '\n'
            f.write(s)



def log_per_batch(tensorboard, loss, batch_number, other_losses=None):
    if tensorboard is not None:
        tensorboard.add_scalar('batch_loss', loss, batch_number)
    if other_losses is not None:
        for j, l in enumerate(other_losses):
            tensorboard.add_scalar(f'batch_loss_{j}', l, batch_number)



def train_ground_state_loop(i, conformation, optimizer, model, loss_function, tensorboard, lr_scheduler, total_batches):
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            true_energies = conformation['energies'].to(device).float()
    
            # zero gradients in the parameter tensors
            optimizer.zero_grad()
    
            # Forwards + backwards + optimize (every batch)
            _, predicted_energies = model((species, coordinates))
            loss = loss_function(predicted_energies, true_energies, species)
            loss.backward()
            optimizer.step()
            
            # Log per batch info
            log_per_batch(tensorboard, loss, lr_scheduler.last_epoch * total_batches + i)

def train_excited_state_loop(i, conformation, optimizer, model, loss_function, tensorboard, lr_scheduler, total_batches):
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            # target ground_energies is of shape (C, )
            target_ground = conformation['energies'].to(device).float()
            # target excited energies is of shape (C, 10)
            target_excited = conformation['energies_ex'].to(device).float()
            target_energies = torch.cat((target_ground.reshape(-1, 1), target_excited), dim=-1)
            # zero gradients in the parameter tensors
            optimizer.zero_grad()
    
            # Forwards + backwards + optimize (every batch)
            # with one energy predicted energies is if shape (C, )
            # with 10 excited energies it is of shape (C, 11) (one ground + 10 excited)
            _, predicted_energies = model((species, coordinates))
            # I need a loss function for excited state energies
            loss = loss_function(predicted_energies, target_energies, species)
            loss.backward()
            optimizer.step()
            
            # Log per batch info
            log_per_batch(tensorboard, loss, lr_scheduler.last_epoch * total_batches + i)


def train_ground_state_energies(model, optimizer, lr_scheduler, loss_function,
        dataset, output_paths, use_tqdm=False, max_epochs=None,
        early_stopping_lr=0.0, config=None):
    max_epochs = max_epochs if max_epochs is not None else maxsize 

    # If the model is already trained, just exit, else, train
    if lr_scheduler.last_epoch == max_epochs: 
        print(f'Model fully trained, with {max_epochs} epochs')
        exit()

    training = datasets.training
    validation = datasets.validation

    # Tensorboard logging, allow toggle
    if output_paths.tensorboard is not None:
        tensorboard = torch.utils.tensorboard.SummaryWriter(output_paths.tensorboard.as_posix())
    else:
        tensorboard = None

    # get the device from an arbitrary tensor on the model
    device = model.aev_computer.ShfR.device

    # at the beginning the epoch is the 0th epoch, epoch 0 means not doing nothing
    # afterwards rmse and logging is done after an epoch finishes
    # initial log
    if  lr_scheduler.last_epoch == 0:
        main_metric, metrics = validate_energies(model, validation)
        log_per_epoch(tensorboard, main_metric, lr_scheduler, optimizer, csv_file=output_paths.csv, metrics=metrics)
    
    total_batches = len(training) 
    print(f"Training starting from epoch {lr_scheduler.last_epoch + 1}")
    print(f"Dataset has {total_batches} batches")
    for _ in range(lr_scheduler.last_epoch, max_epochs):
        start = time.time()

        # Training loop per epoch
        if use_tqdm: 
            conformations = tqdm(enumerate(training), total=total_batches, desc=f'epoch {lr_scheduler.last_epoch}')
        else:
            conformations = enumerate(training)
        for i, conformation in conformations:
            species = conformation['species'].to(device)
            coordinates = conformation['coordinates'].to(device).float()
            true_energies = conformation['energies'].to(device).float()
    
            # zero gradients in the parameter tensors
            optimizer.zero_grad()
    
            # Forwards + backwards + optimize (every batch)
            _, predicted_energies = model((species, coordinates))
            loss = loss_function(predicted_energies, true_energies, species)
            loss.backward()
            optimizer.step()
            
            # Log per batch info
            log_per_batch(tensorboard, loss, lr_scheduler.last_epoch * total_batches + i)
    
        # Validate (every epoch)
        main_metric, metrics = validate_energies(model, validation)
    
        # Step the scheduler and save the best model (every epoch)
        if lr_scheduler.is_better(main_metric.values()[0], lr_scheduler.best):
            save_checkpoint(output_paths.best, model, optimizer, lr_scheduler)
        lr_scheduler.step(main_metric.values()[0])

        end = time.time()
    
        # Log per epoch info
        log_per_epoch(tensorboard, main_metric, metrics, lr_scheduler, optimizer, end - start, csv_file=output_paths.csv, metrics=metrics)

        # Checkpoint
        save_checkpoint(output_paths.latest, model, optimizer, lr_scheduler)
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < early_stopping_lr:
            print('Training finishing due to early stopping lr')
            break
    
    # save all hparams that are either floats or ints to tensorboard
    hparams_dict = {}
    for key in config.keys():
        hparams_dict.update({f'{key}/{k}': v for k, v in config[key].items() if isinstance(v, (float, int))})
    
    # the hparams/ is necessary for the metrics
    metrics = {f'hparams/{k}' : v for k, v in metrics.items()}
    main_key = main_metric.keys()[0]
    metrics.update({f'hparams/best_{main_key}' : lr_scheduler.best})
    tensorboard.add_hparams(hparams_dict, metrics)
    print('Training finished')

def update_scan_search_config(config, scan_search, idx):
    for parameter, range_ in scan_search.items():
        try:
            new_value = range_[idx]
        except IndexError as e:
            print('Attempted a scan search but search is already done')
            raise e
        insert_in_key(config, parameter, new_value)
    return config

def update_random_search_config(config, random_search):
    for parameter, range_ in random_search.items():
        new_value = get_one_random_instance(parameter, range_)
        insert_in_key(config, parameter, new_value)

def make_output_path_trial_dir(output_paths, yaml_name):
    # function that creates a trial dir either for a random search or for a scan search
    output_paths = output_paths.joinpath(yaml_name)
    idx = 0
    # make a new directory with a trial index
    while True:
        try:
            output_paths.joinpath(f'trial_{idx}').mkdir(parents=True)
            output_paths = output_paths.joinpath(f'trial_{idx}')
            break
        except FileExistsError:
            idx += 1
    return output_paths, idx

def load_training_configuration(yaml_path):
    if isinstance(yaml_path, str):
        yaml_path = Path(args.yaml_path).resolve()
    random_search = dict()
    scan_search = dict()
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    yaml_name = yaml_path.stem
    if yaml_name[-6:] == 'random':
        original_config = copy.deepcopy(config)
        random_search = config.pop('random_search')
    elif yaml_name[-4:] == 'scan':
        original_config = copy.deepcopy(config)
        scan_search = config.pop('scan_search')
    else:
        assert 'random_search' not in config
        assert 'scan_search' not in config
    return original_config, config, random_search, scan_search, yaml_name

def get_output_paths(output_paths_raw, yaml_name):
    if args.output_paths is None:
        # by default this is saved into a very nice folder labeled with the name
        # of the yaml file, inside training_template
        output_paths = Path(__file__).resolve().parent.parent.parent.joinpath('training_outputs/')
        # if the name ends with search, assume this is a hyperparameter search
        if random_search or scan_search:
            output_paths, idx = make_output_path_trial_dir(output_paths, yaml_name)
        else:
            output_paths = output_paths.joinpath(yaml_name)
            output_paths.mkdir(parents=True, exist_ok=True)
    else:
        output_paths = Path(args.output_paths).resolve()
    return output_paths, idx

def dump_yaml_input(output_paths, yaml_name, random_search, scan_search):
    yaml_output = output_paths.joinpath(f'{yaml_name}.yaml')
    with open(yaml_output, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # dump the original configuration in the parent folder
    # if performing a random hyperparameter search
    if random_search or scan_search:
        yaml_original_output = output_paths.parent.joinpath(f'{yaml_name}_original.yaml')
        if not yaml_original_output.is_file():
            with open(yaml_original_output, 'w') as f:
                yaml.dump(original_config, f, sort_keys=False)

def load_datasets(config, output_paths, dataset_path_raw, model):
    Datasets = namedtuple('Datasets', 'training validation test')
    # fetch dataset path from input arguments or from configuration file 
    if output_paths.dataset_pkl.is_file():
        print('Unpickling training and validation files located inside output directory')
        with open(output_paths.dataset_pkl, 'rb') as f:
            pickled_dataset = pickle.load(f)
            training = pickled_dataset['training']
            validation = pickled_dataset['validation']
    else:
        # case when pickled files don't exist in the output path
        if args.dataset_path is not None:
            data_path = Path(args.dataset_path).resolve()
        else:
            data_path = Path(config['datasets']['dataset_path']).resolve()
        assert data_path.is_file()

        if data_path.suffix == '.h5':
            print('Loading training and validation from h5 file')
            data.PROPERTIES = tuple(v for k, v in config['datasets']['nonstandard_keys'].items() if k not in ['species', 'coordinates'])
            training, validation = data.load(data_path.as_posix(), nonstandard_keys=config['datasets']['nonstandard_keys'])\
                            .subtract_self_energies(model.energy_shifter, model.species_order())\
                            .species_to_indices('periodic_table')\
                            .shuffle().split(config['datasets']['split_percentage'], None)

            training = training.collate(config['datasets']['batch_size']).cache()
            validation = validation.collate(config['datasets']['batch_size']).cache()
            with open(output_paths.dataset_pkl, 'wb') as f:
                pickled_dataset = {'training': training, 'validation':validation}
                pickle.dump(pickled_dataset, f)
            print('\n')
        elif data_path.suffix == '.pkl':
            print('Unpickling external training and validation files')
            with open(data_path, 'rb') as f:
                pickled_dataset = pickle.load(f)
                training = pickled_dataset['training']
                validation = pickled_dataset['validation']
    training = training.pin_memory()
    validation = validation.pin_memory()

    datasets = Datasets(training=training, validation=validation, test=None)
    return datasets
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a network from a yaml file configuration')
    parser.add_argument('-y', '--yaml-path', type=str, required=True, help='Input yaml configuration')
    parser.add_argument('-d', '--dataset-path', default=None, help='Path to the dataset to train on')
    parser.add_argument('-o', '--output-paths', default=None, help='Path for'
            ' tensorboard, latest and best checkpoints, also holds pickled'
            ' datasets after loading, if the dataset path is an h5 file')
    args = parser.parse_args()

    # Get the configuration for the training
    original_config, config, random_search, scan_search, yaml_name = load_training_configuration(args.yaml_path)

    output_paths, idx = get_output_paths(args.output_paths, yaml_name)
    
    # move to the next available parameter for all values in the scan or random
    # search, in the case of the random search a random parameter from a 
    # range is obtained
    if scan_search:
        update_scan_search_config(config, scan_search, idx)

    if random_search:
        update_random_search_config(config, random_search)

    # Paths for output (tensorboard, checkpoints and validation / training pickles)
    dump_yaml_input(output_paths, yaml_name, random_search, scan_search)
    latest_path = output_paths.joinpath('latest.pt')
    best_path = output_paths.joinpath('best.pt')
    csv_path = output_paths.joinpath('log.csv')

    # use the parent folder as a store for the pickled datasets, TODO:
    # WARNING! watch out, this can create issues if many processes are
    # started simultaneously and the datasets have not been pickled yet, 
    # in that case all processes will start to create different pickled
    # splits and they will overwrite the files each time one finishes
    if random_search or scan_search:
        dataset_pkl_path = output_paths.parent.joinpath('dataset.pkl')
    else:
        dataset_pkl_path = output_paths.joinpath('dataset.pkl')

    tensorboard_path = output_paths
    OutputPaths = namedtuple('OutputPaths', 'tensorboard best latest dataset_pkl csv')
    output_paths = OutputPaths(best=best_path, latest=latest_path,
            tensorboard=tensorboard_path, dataset_pkl=dataset_pkl_path, csv=csv_path)
    
    # setup model and initialize parameters
    # setting shift before output to false makes the model NOT add saes before output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemplateModel.from_yaml(config).to(device).shift_before_output_(False)
    init_function = getattr(torchani.training, config['init']['function'])
    model.apply(init_function)

    # setup optimizer, scheduler and loss
    Optimizer = getattr(optim, config['optimizer'].pop('class'))
    LrScheduler = getattr(lr_scheduler, config['lr_scheduler'].pop('class'))
    LossFunction = getattr(torchani.training, config['loss'].pop('class'))

    optimizer = Optimizer(model.parameters(), **config['optimizer'])
    lr_scheduler = LrScheduler(optimizer, **config['lr_scheduler'])
    loss_function = LossFunction()
   
    # Logic for loading datasets, setup training and validation sets and add
    # stuff to EnergyShifter
    datasets = load_datasets(config, output_paths, args.dataset_path, model)
    
    # load model parameters from checkpoint if it exists
    load_from_checkpoint(output_paths.latest, model, optimizer, lr_scheduler)
     
    # train model 
    train_ground_state_energies(model, 
            optimizer, 
            lr_scheduler, 
            loss_function, 
            datasets,
            output_paths, 
            **config['general'], config=config)



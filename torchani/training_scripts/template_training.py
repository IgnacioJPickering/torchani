from pathlib import Path
from collections import namedtuple
from sys import maxsize
import pickle
import time

import torch
import yaml
import torch.utils.tensorboard
from tqdm import tqdm

import torchani
from torchani import data
from torchani.modules import TemplateModel
from torchani.training import validate_energies, init_traditional

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

def log_per_epoch(tensorboard, validation_rmse, lr_scheduler, optimizer, epoch_time=None, verbose=True):
    if verbose:
        print(f'Validation RMSE: {validation_rmse} after epoch {lr_scheduler.last_epoch} time: {epoch_time if epoch_time is not None else 0}')
    if tensorboard is not None:
        epoch_number = lr_scheduler.last_epoch
        tensorboard.add_scalar('validation_rmse', validation_rmse, epoch_number)
        tensorboard.add_scalar('best_validation_rmse', lr_scheduler.best, epoch_number)
        tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'] , epoch_number)
        if epoch_time is not None:
            tensorboard.add_scalar('epoch_time', epoch_time, epoch_number)

def log_per_batch(tensorboard, loss, batch_number):
    if tensorboard is not None:
        tensorboard.add_scalar('batch_loss', loss, batch_number)

def train(model, optimizer, lr_scheduler, loss_function, dataset, output_paths, use_tqdm=False, max_epochs=None, early_stopping_lr=0.0):
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
        validation_rmse = validate_energies(model, validation)
        log_per_epoch(tensorboard, validation_rmse, lr_scheduler, optimizer)
    
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
        validation_rmse = validate_energies(model, validation)
    
        # Step the scheduler and save the best model (every epoch)
        if lr_scheduler.is_better(validation_rmse, lr_scheduler.best):
            save_checkpoint(output_paths.best, model, optimizer, lr_scheduler)
        lr_scheduler.step(validation_rmse)

        end = time.time()
    
        # Log per epoch info
        log_per_epoch(tensorboard, validation_rmse, lr_scheduler, optimizer, end - start)

        # Checkpoint
        save_checkpoint(output_paths.latest, model, optimizer, lr_scheduler)
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < early_stopping_lr:
            break
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a network from a yaml file configuration')
    parser.add_argument('-d', '--dataset-path', default=None, help='Path to the dataset to train on')
    parser.add_argument('-y', '--yaml-path', type=str, default='../../training_templates/hyper.yaml', help='Input yaml configuration')
    parser.add_argument('-o', '--output-paths', type=str, default='.', help='Path for'
            ' tensorboard, latest and best checkpoints, also holds pickled datasets after loading')
    args = parser.parse_args()

    # Paths for output (tensorboard, checkpoints and validation / training pickles)
    output_paths = Path(args.output_paths).resolve()
    latest_path = output_paths.joinpath('latest.pt')
    best_path = output_paths.joinpath('best.pt')
    training_pkl = output_paths.joinpath('training.pkl')
    validation_pkl = output_paths.joinpath('validation.pkl')
    tensorboard_path = output_paths
    OutputPaths = namedtuple('OutputPaths', 'tensorboard best latest training_pkl validation_pkl')
    output_paths = OutputPaths(best=best_path, latest=latest_path, tensorboard=tensorboard_path, training_pkl=training_pkl, validation_pkl=validation_pkl)

    # Yaml file Path
    yaml_path = Path(args.yaml_path).resolve()

    # three paths, path to the yaml file, and paths to outputs
    # Get the configuration for the training
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # setup model and initialize parameters
    # setting shift before output to false makes the model NOT add saes before output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemplateModel.from_yaml(config).to(device).shift_before_output_(False)
    model.apply(init_traditional)

    # setup optimizer, scheduler and loss
    optimizer = getattr(optim, config['optimizer']['class'])(model.parameters(), **config['optimizer']['kwargs'])
    lr_scheduler = getattr(lr_scheduler, config['lr_scheduler']['class'])(optimizer, **config['lr_scheduler']['kwargs'])
    loss_function = getattr(torchani.training, config['loss']['class'])()
    
    # setup training and validation sets
    Datasets = namedtuple('Datasets', 'training validation test')
    
    # fetch dataset path from input arguments or from configuration file 
    pkl_files_exist = (output_paths.training_pkl.is_file(), output_paths.validation_pkl.is_file())
    if all(pkl_files_exist):
        print('Unpickling training and validation files')
        with open(output_paths.training_pkl, 'rb') as f:
            training = pickle.load(f)
        with open(output_paths.validation_pkl, 'rb') as f:
            validation = pickle.load(f)
    elif pkl_files_exist[0] != pkl_files_exist[1]:
        raise RuntimeError('Only one of the validation / training files exist, not possible to train')
    else:
        # case when none of the pickle files exist
        print('Loading training and validation from h5 file')
        if args.dataset_path is not None:
            data_path = Path(args.dataset_path).resolve()
        else:
            data_path = Path(config['datasets']['dataset_path']).resolve()
        assert data_path.is_file()
        data.PROPERTIES = tuple(v for k, v in config['datasets']['nonstandard_keys'].items() if k not in ['species', 'coordinates'])
        training, validation = data.load(data_path.as_posix(), nonstandard_keys=config['datasets']['nonstandard_keys'])\
                        .subtract_self_energies(model.energy_shifter, model.species_order())\
                        .species_to_indices('periodic_table')\
                        .shuffle().split(config['datasets']['split_percentage'], None)

        training = training.collate(config['datasets']['batch_size']).cache()
        validation = validation.collate(config['datasets']['batch_size']).cache()
        with open(output_paths.training_pkl, 'wb') as f:
            pickle.dump(training, f)
        with open(output_paths.validation_pkl, 'wb') as f:
            pickle.dump(validation, f)
        print('\n')
    training = training.pin_memory()
    validation = validation.pin_memory()

    datasets = Datasets(training=training, validation=validation, test=None)
    
    # load model parameters from checkpoint if it exists
    load_from_checkpoint(output_paths.latest, model, optimizer, lr_scheduler)

    # train model 
    train(model, 
            optimizer, 
            lr_scheduler, 
            loss_function, 
            datasets,
            output_paths, 
            **config['general'])

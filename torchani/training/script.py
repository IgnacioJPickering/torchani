from pathlib import Path
from collections import namedtuple
from sys import maxsize
import pickle
import time
import copy
import random

import torch
import yaml
import torch.utils.tensorboard
from tqdm import tqdm

import torchani
from torchani import data
from torchani.modules import TemplateModel
from torchani.training import get_one_random_instance, insert_in_key, move_settings_to_index

from torch import optim
from torch.optim import lr_scheduler

def load_from_checkpoint(latest_checkpoint, model, optimizer, lr_scheduler, loss_function, other_scheduler=None):
    if latest_checkpoint.is_file():
        checkpoint = torch.load(latest_checkpoint.as_posix())
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if other_scheduler is not None:
            other_scheduler.load_state_dict(checkpoint['other_scheduler'])
        loss_dict = checkpoint.get('loss_function', None) 
        if loss_dict is not None:
            loss_function.load_state_dict(loss_dict)

class Trainer:

    def __init__(self, model, optimizer, lr_scheduler, loss_function,
            validation_function, output_paths, hparams, other_scheduler=None, verbose=True):
        
        # model, optimizer, lr_scheduler, loss and validation function are
        # custom
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.validate = validation_function
        self.other_scheduler = other_scheduler

        self.output_paths = output_paths
        self.hparams = hparams

        # device is obtained from an arbitrary tensor
        self.device = model.aev_computer.ShfR.device
        self.verbose = verbose

        # Tensorboard logging, allow toggle
        if output_paths.tensorboard is not None:
            self.tensorboard = torch.utils.tensorboard.SummaryWriter(output_paths.tensorboard.as_posix())
        else:
            self.tensorboard = None
        
    def ground_state_loop(self, batch_number, conformation):
       species = conformation['species'].to(self.device, non_blocking=True)
       coordinates = conformation['coordinates'].to(self.device, non_blocking=True).float()
       true_energies = conformation['energies'].to(self.device, non_blocking=True).float()
    
       # zero gradients in the parameter tensors
       self.optimizer.zero_grad()
    
       # Forwards + backwards + optimize (every batch)
       _, predicted_energies = self.model((species, coordinates))
       loss = self.loss_function(predicted_energies, true_energies, species)
       loss.backward()
       self.optimizer.step()
       
       # Log per batch info
       if self.log_every_batch:
           self.log_per_batch(loss, batch_number)
           return None, None
       else:
           return loss, None

    def excited_states_and_foscs_loop(self, batch_number, conformation):
        # train against energies and ex dipoles (or foscs or sqdipoles), validate 
        # on fosc

        species = conformation['species'].to(self.device, non_blocking=True)
        coordinates = conformation['coordinates'].to(self.device, non_blocking=True).float()
        # target ground_energies is of shape (C, )
        target_ground = conformation['energies'].to(self.device, non_blocking=True).float()
        # target excited energies is of shape (C, 10)
        target_ex = conformation['energies_ex'].to(self.device, non_blocking=True).float()

        if self.foscs_only:
            other_ex = conformation['foscs_ex'].to(self.device, non_blocking=True).float()
        elif self.sqdipoles_only:
            other_ex = conformation['sqdipoles_ex'].to(self.device, non_blocking=True).float()
        else:
            other_ex = conformation['dipoles_ex'].to(self.device, non_blocking=True).float().permute(0, 2, 1) 

        target_energies = torch.cat((target_ground.reshape(-1, 1), target_ex), dim=-1)
        # zero gradients in the parameter tensors
        self.optimizer.zero_grad()
        
        # Forwards + backwards + optimize (every batch)
        # with one energy predicted energies is if shape (C, )
        # with 10 excited energies it is of shape (C, 11) (one ground + 10 excited)

        # I will only predict ex dipoles or sqdipoles or foscs
        _, predicted_energies, predicted_other_ex = self.model((species, coordinates))

        # I need a loss function for excited state energies
        # note that this loss takes in energies AND dipoles!
        loss, losses = self.loss_function(predicted_energies, target_energies, predicted_other_ex, other_ex, species)
        loss.backward()
        self.optimizer.step()
        
        # Log per batch info
        if self.log_every_batch:
            self.log_per_batch(loss, batch_number, other=losses)
            return None, None
        else:
            return loss, losses
    
    def excited_state_loop(self, batch_number, conformation):
        species = conformation['species'].to(self.device, non_blocking=True)
        coordinates = conformation['coordinates'].to(self.device, non_blocking=True).float()
        # target ground_energies is of shape (C, )
        target_ground = conformation['energies'].to(self.device, non_blocking=True).float()
        # target excited energies is of shape (C, 10)
        target_excited = conformation['energies_ex'].to(self.device, non_blocking=True).float()
        target_energies = torch.cat((target_ground.reshape(-1, 1), target_excited), dim=-1)
        # zero gradients in the parameter tensors
        self.optimizer.zero_grad()
        
        # Forwards + backwards + optimize (every batch)
        # with one energy predicted energies is if shape (C, )
        # with 10 excited energies it is of shape (C, 11) (one ground + 10 excited)
        _, predicted_energies = self.model((species, coordinates))
        # I need a loss function for excited state energies
        loss, losses = self.loss_function(predicted_energies, target_energies, species)
        loss.backward()
        self.optimizer.step()
        
        # Log per batch info
        if self.log_every_batch:
            self.log_per_batch(loss, batch_number, other=losses)
            return None, None
        else:
            return loss, losses

    def train(self, datasets, use_tqdm=False, max_epochs=maxsize, early_stopping_lr=0.0, loop='ground_state_loop', log_every_batch=False, foscs_only=False, sqdipoles_only=False):
        self.foscs_only = foscs_only
        self.sqdipoles_only = sqdipoles_only
        self.log_every_batch = log_every_batch
        # If the model is already trained, just exit, else, train
        if self.lr_scheduler.last_epoch == max_epochs: 
            if self.verbose: print(f'Model fully trained, with {max_epochs} epochs')
            exit(0)

        training = datasets.training
        validation = datasets.validation

        # at the beginning the epoch is the 0th epoch, epoch 0 means not doing nothing
        # afterwards rmse and logging is done after an epoch finishes
        # initial log
        if  self.lr_scheduler.last_epoch == 0:
            main_metric, metrics = self.validate(self.model, validation)
            self.log_per_epoch(main_metric, metrics=metrics)
        
        total_batches = len(training) 
        if self.verbose: print(f"Training starting from epoch"
                               f" {self.lr_scheduler.last_epoch + 1}\n"
                               f"Dataset has {total_batches} batches")

        # avoid stupidity (?) in data
        if isinstance(training, data.TransformableIterable):
            training = list(training)

        for _ in range(self.lr_scheduler.last_epoch, max_epochs):
            start = time.time()
            
            # shuffling batches
            random.shuffle(training) 
            
            # Setup tqdm if necessary
            if use_tqdm: 
                conformations = tqdm(enumerate(training), total=total_batches, desc=f'epoch {self.lr_scheduler.last_epoch}')
            else:
                conformations = enumerate(training)

            # Perform inner training loop
            training_loop = getattr(self, loop)


            for i, conformation in conformations:
                batch_number = i + self.lr_scheduler.last_epoch * total_batches
                loss, other_losses = training_loop(batch_number, conformation)
        
            # Validate (every epoch)
            main_metric, metrics = self.validate(self.model, validation)
        
            # Step the scheduler and save the BEST model (every epoch)
            main_value = list(main_metric.values())[0]
            if self.lr_scheduler.is_better(main_value, self.lr_scheduler.best):
                self.save_checkpoint(self.output_paths.best)

            self.lr_scheduler.step(main_value)
            if self.other_scheduler is not None:
                self.other_scheduler.step()

            # Save LATEST Checkpoint
            self.save_checkpoint(self.output_paths.latest)

            end = time.time()

            # Log per epoch info
            self.log_per_epoch(main_metric, metrics=metrics, epoch_time=end - start, loss=loss, other_losses=other_losses)
            
            # Early stopping
            if self.optimizer.param_groups[0]['lr'] < early_stopping_lr:
                if self.verbose: print('Training finishing due to early stopping lr')
                break

        self.log_hparams(self.hparams, main_metric, metrics)
        if self.verbose: print('Training finished')

    def log_hparams(self, hparams, main_metric, metrics):
        if self.tensorboard is not None: 
            # the hparams/ is necessary for the metrics
            if metrics is not None:
                metrics = {f'hparams/{k}' : v for k, v in metrics.items()}
            else:
                metrics = dict()
            (main_key, main_value), = main_metric.items()
            metrics.update({f'hparams/{main_key}' : main_value})
            metrics.update({f'hparams/best_{main_key}' : self.lr_scheduler.best})
            self.tensorboard.add_hparams(hparams, metrics)

    def save_checkpoint(self, path):
        dictionary = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        if self.other_scheduler is not None:
            dictionary.update({'other_scheduler': self.other_scheduler.state_dict()})
        if list(self.loss_function.parameters()):
            dictionary.update({'loss_function': self.loss_function.state_dict()})
        torch.save(dictionary
                , path.as_posix())

    def log_per_batch(self, loss, batch_number, other=None):
        if self.tensorboard is not None:
            self.tensorboard.add_scalar('batch_loss', loss, batch_number)
        if other is not None:
            for j, l in enumerate(other):
                self.tensorboard.add_scalar(f'batch_loss_{j}', l, batch_number)

    def log_per_epoch(self, main_metric, metrics=None, epoch_time=None, loss=None, other_losses=None):
        # Other metrics is a dictionary with different metrics
        (main_key, main_value), = main_metric.items()
        if epoch_time is None:
            epoch_time = 0.0
        epoch_number = self.lr_scheduler.last_epoch
        if self.verbose:
            print(f'{main_key} {main_value} after epoch {self.lr_scheduler.last_epoch} time: {epoch_time if epoch_time is not None else 0}')
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(f'{main_key}', main_value, epoch_number)
            self.tensorboard.add_scalar(f'best_{main_key}', self.lr_scheduler.best, epoch_number)
            self.tensorboard.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'] , epoch_number)
            if self.hparams.get('aev_computer/trainable_etas'):
                for j, v in enumerate(self.model.aev_computer.EtaR):
                    self.tensorboard.add_scalar(f'aev/EtaR_{j}', v, epoch_number)
                for j, v in enumerate(self.model.aev_computer.EtaA.view(-1)):
                    self.tensorboard.add_scalar(f'aev/EtaA_{j}', v, epoch_number)
            if self.hparams.get('aev_computer/trainable_shifts') or self.hparams.get('aev_computer/trainable_radial_shifts'):
                for j, v in enumerate(self.model.aev_computer.ShfR.view(-1)):
                    self.tensorboard.add_scalar(f'aev/ShfR_{j}', v, epoch_number)
            if self.hparams.get('aev_computer/trainable_shifts') or self.hparams.get('aev_computer/trainable_angular_shifts'):
                for j, v in enumerate(self.model.aev_computer.ShfA.view(-1)):
                    self.tensorboard.add_scalar(f'aev/ShfA_{j}', v, epoch_number)
            if self.hparams.get('aev_computer/trainable_shifts') or self.hparams.get('aev_computer/trainable_angle_sections'):
                for j, v in enumerate(self.model.aev_computer.ShfZ.view(-1)):
                    self.tensorboard.add_scalar(f'aev/ShfZ_{j}', v, epoch_number)
            if metrics is not None:
                for k, v in metrics.items():
                    self.tensorboard.add_scalar(f'{k}', v, epoch_number)
            if epoch_time is not None:
                self.tensorboard.add_scalar('epoch_time', epoch_time, epoch_number)
            if loss is not None:
                self.tensorboard.add_scalar('epoch_loss', loss, epoch_number)
            if other_losses is not None:
                for j, l in enumerate(other_losses):
                    self.tensorboard.add_scalar(f'epoch_loss_{j}', l, epoch_number)
        if self.output_paths.csv is not None:
            if not self.output_paths.csv.is_file():
                with open(self.output_paths.csv, 'w') as f:
                    s = f'{main_key} epoch time lr'
                    if metrics is not None:
                        for k in metrics.keys():
                            s += f' {k}'
                    s += '\n'
                    f.write(s)
    
            with open(self.output_paths.csv, 'a') as f:
                s = f'{main_value} {epoch_number} {epoch_time}'
                if metrics is not None:
                    for v in metrics.values():
                        s += f' {v}'
                s += '\n'
                f.write(s)

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

def load_global_configuration():
    global_config_path = Path(__file__).resolve().parent.parent.parent.joinpath('training_templates/global_config.yaml')
    with open(global_config_path, 'r') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    return global_config

def load_configuration(yaml_path):
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
        original_config = None
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
            idx = None
            output_paths = output_paths.joinpath(yaml_name)
            output_paths.mkdir(parents=True, exist_ok=True)
    else:
        idx = None
        output_paths = Path(args.output_paths).resolve()
    return output_paths, idx

def dump_yaml_input(output_paths, yaml_name, original_config):
    yaml_output = output_paths.joinpath(f'{yaml_name}.yaml')
    with open(yaml_output, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # dump the original configuration in the parent folder
    # if performing a random hyperparameter search
    if original_config is not None:
        # We have either a random or a scan search
        yaml_original_output = output_paths.parent.joinpath(f'{yaml_name}_original.yaml')
        if not yaml_original_output.is_file():
            with open(yaml_original_output, 'w') as f:
                yaml.dump(original_config, f, sort_keys=False)

def load_datasets(ds_config, output_paths, dataset_path_raw, model, global_config=None):
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
            if global_config is None:
                data_path = Path(ds_config['dataset_path']).resolve()
            else:
                data_path = Path(global_config['datasets']).resolve().joinpath(ds_config['dataset_path'])
        assert data_path.is_file() or data_path.is_dir()

        if data_path.suffix == '.h5' or data_path.is_dir():
            print('Loading training and validation from h5 file or directory of h5 files')
            data.PROPERTIES = tuple(v for k, v in ds_config['nonstandard_keys'].items() if k not in ['species', 'coordinates'])
            # also subtract self energies for ex energies if they are present
            # this may need to be modified in the future
            ex_key = 'energies_ex' if 'energies_ex' in data.PROPERTIES else None
            training, validation = data.load(data_path.as_posix(), nonstandard_keys=ds_config['nonstandard_keys'])\
                            .subtract_self_energies(model.energy_shifter, model.species_order(), ex_key=ex_key)\
                            .species_to_indices('periodic_table')\
                            .shuffle().split(ds_config['split_percentage'], None)

            print('Self energies: ')
            print(model.energy_shifter.self_energies)

            training = training.collate(ds_config['batch_size']).cache()
            validation = validation.collate(ds_config['batch_size']).cache()
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
    parser = argparse.ArgumentParser(
                   description='Train a network from a yaml file configuration')
    parser.add_argument('yaml_path', type=str, help='Input yaml configuration')
    parser.add_argument('-d', '--dataset-path', default=None, help='Path to the'
                        ' dataset to train on')
    parser.add_argument('-o', '--output-paths', default=None, help='Path for'
            ' tensorboard, latest and best checkpoints, also holds pickled'
            ' datasets after loading, if the dataset path is an h5 file')
    args = parser.parse_args()

    # Yaml parsing: Get the configuration from the yaml file
    original_config, config, random_search, scan_search, yaml_name = \
                                     load_configuration(args.yaml_path)
    global_config = load_global_configuration()

    output_paths, idx = get_output_paths(args.output_paths, yaml_name)
    
    # move to the next available parameter for all values in the scan or random
    # search, in the case of the random search a random parameter from a 
    # range is obtained
    if scan_search:
        # idx is needed for this
        assert idx is not None
        move_settings_to_index(config, scan_search, idx)

    if random_search:
        # idx is not needed here but must have been calculated
        assert idx is not None  
        update_random_search_config(config, random_search)

    # Paths for output (tensorboard, checkpoints and validation / training pickles)
    dump_yaml_input(output_paths, yaml_name, original_config)
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
    model.train()
    init_function = getattr(torchani.training, config['init']['function'])
    model.apply(init_function)

    # setup optimizer, scheduler , loss and validation
    # validation function must take in model and validation set, as
    # arguments, and output main_metric, metrics
    Optimizer = getattr(optim, config['optimizer'].pop('class'))
    LrScheduler = getattr(lr_scheduler, config['lr_scheduler'].pop('class'))
    LossFunction = getattr(torchani.training, config['loss'].pop('class'))

    loss_function = LossFunction(**config['loss']).to(device)

    model_params = dict(params= model.neural_networks.parameters(), **config['optimizer'])
    optimizer_configuration = [model_params]

    if list(model.aev_computer.parameters()):
        aev_params = dict(params= model.aev_computer.parameters(), **config['optimizer_aev'])
        optimizer_configuration.append(aev_params)

    if list(loss_function.parameters()):
        loss_params = dict(params= loss_function.parameters(), **config['optimizer_loss'])
        optimizer_configuration.append(loss_params)
    
    optimizer = Optimizer(optimizer_configuration)
    scheduler = LrScheduler(optimizer, **config['lr_scheduler'])
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler = scheduler
        other_scheduler = None
    else:
        # keep RLROP only to keep track of best rmse
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=99999999, threshold=0)
        other_scheduler = scheduler


    validation_function = getattr(torchani.training, config.pop('validation_function'))
   
    # Logic for loading datasets, setup training and validation sets and add
    # stuff to EnergyShifter
    datasets = load_datasets(config['datasets'], output_paths, args.dataset_path, model, global_config=global_config)
    torch.cuda.empty_cache()
    
    # load model parameters from checkpoint if it exists
    load_from_checkpoint(output_paths.latest, model, optimizer, lr_scheduler, loss_function, other_scheduler=other_scheduler)

    # save all hparams that are either floats or ints to tensorboard
    hparams = {}
    for key in config.keys():
        hparams.update({f'{key}/{k}': v for k, v in config[key].items() if isinstance(v, (float, int))})


    trainer = Trainer(model, optimizer, lr_scheduler, loss_function, validation_function, output_paths, hparams, other_scheduler=other_scheduler)
    trainer.train(datasets, **config['general'],
            use_tqdm=global_config['use_tqdm'],
            log_every_batch=global_config['log_every_batch'])
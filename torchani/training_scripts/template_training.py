from pathlib import Path
 
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

# get device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the configuration for the training
yaml_path = '../training/hyper.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# setup model and initialize parameters
# this uses H C N O species and by default has periodic_table_index on
# setting shift before output to false makes the model NOT add saes before output
model = TemplateModel.from_yaml(config).to(device).shift_before_output_(False)
model.apply(init_traditional)

# Some configuration variables, which are general
batch_size = config['batch_size'] 
max_epochs = config['max_epochs']
early_stopping_lr = config['early_stopping_lr'] 

# setup training and validation sets
data_path = Path(__file__).resolve().parent.joinpath('../../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
assert data_path.is_file()
training, validation = data.load(data_path.as_posix()).subtract_self_energies(model.energy_shifter, model.species_order()).species_to_indices('periodic_table').shuffle().split(0.8, None)
training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('\n')

# setup optimizer, scheduler and loss
optimizer = getattr(optim, config['optimizer']['class'])(model.parameters(), **config['optimizer']['kwargs'])
lr_scheduler = getattr(lr_scheduler, config['lr_scheduler']['class'])(optimizer, **config['lr_scheduler']['kwargs'])
loss_function = getattr(torchani.training, config['loss']['class'])()

# Tensorboard logging
tensorboard_path = Path(__file__).resolve().parent
tensorboard = torch.utils.tensorboard.SummaryWriter(tensorboard_path.as_posix())

# Paths for checkpoints
best_model_checkpoint = Path(__file__).resolve().parent.joinpath('best.pt')
latest_checkpoint = Path(__file__).resolve().parent.joinpath('latest.pt')

if latest_checkpoint.is_file():
    checkpoint = torch.load(latest_checkpoint.as_posix())
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

# If the model is already trained, just exit
if lr_scheduler.last_epoch == max_epochs: 
    print('Model fully trained, with {max_epochs} epochs')
    exit()

# at the beginning the epoch is the 0th epoch, epoch 0 means not doing nothing
# afterwards rmse and logging is done after an epoch finishes

# initial log
if  lr_scheduler.last_epoch == 0:
    validation_rmse = validate_energies(model, validation, device)
    print(f'Validation RMSE: {validation_rmse} after epoch {lr_scheduler.last_epoch}')
    tensorboard.add_scalar('validation_rmse', validation_rmse, lr_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', lr_scheduler.best, lr_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], lr_scheduler.last_epoch)

total_conformations = len(training) 
print("Training starting from epoch", lr_scheduler.last_epoch + 1)

for _ in range(lr_scheduler.last_epoch, max_epochs):

    # Training loop per epoch
    if config['use_tqdm']: 
        conformations = tqdm(enumerate(training), total=total_conformations, desc=f'epoch {lr_scheduler.last_epoch}')
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
        tensorboard.add_scalar('batch_loss', loss, lr_scheduler.last_epoch * total_conformations + i)

    # Validate (every epoch)
    validation_rmse = validate_energies(model, validation, device)

    # Step the scheduler and save the best model (every epoch)
    if lr_scheduler.is_better(validation_rmse, lr_scheduler.best):
        torch.save(model.state_dict(), best_model_checkpoint.as_posix())
    lr_scheduler.step(validation_rmse)

    # Log per epoch info
    print(f'Validation RMSE: {validation_rmse} after epoch {lr_scheduler.last_epoch}')
    tensorboard.add_scalar('validation_rmse', validation_rmse, lr_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', lr_scheduler.best, lr_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'] , lr_scheduler.last_epoch)
    
    # Early stopping
    if optimizer.param_groups[0]['lr'] < early_stopping_lr:
        break

    # Checkpoint
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }, latest_checkpoint.as_posix())

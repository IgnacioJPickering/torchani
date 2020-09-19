from pathlib import Path

import torch
import torch.utils.tensorboard
from tqdm import tqdm

from torchani import data
from torchani.modules import TemplateModel
from torchani.training import RootAtomsLoss, validate_energies, init_traditional

# get device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup model and initialize parameters
# this uses H C N O species and by default has periodic_table_index on
# setting shift before output to false makes the model NOT add saes before output
model = TemplateModel.like_ani1x().to(device).shift_before_output_(False)
model.apply(init_traditional)

# Some configuration variables
batch_size = 2560
max_epochs = 100
early_stopping_learning_rate = 1.0e-5

# setup training and validation sets
data_path = Path(__file__).resolve().parent.joinpath('../../dataset/ani1-up_to_gdb4/ani_gdb_s01.h5').resolve()
assert data_path.is_file()
training, validation = data.load(data_path.as_posix()).subtract_self_energies(model.energy_shifter, model.species_order()).species_to_indices('periodic_table').shuffle().split(0.8, None)
training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()

# setup optimizer, scheduler, loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.00001, amsgrad=False, eps=1e-8)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, threshold=0)
loss_function = RootAtomsLoss()

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

# initial log
validation_rmse = validate_energies(model, validation, device)
print('RMSE:', validation_rmse, 'at epoch', lr_scheduler.last_epoch + 1)
tensorboard.add_scalar('validation_rmse', validation_rmse, lr_scheduler.last_epoch)
tensorboard.add_scalar('best_validation_rmse', lr_scheduler.best, lr_scheduler.last_epoch)
tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], lr_scheduler.last_epoch)

total_conformations = len(training)
print("Training starting from epoch", lr_scheduler.last_epoch + 1)
for _ in range(lr_scheduler.last_epoch + 1, max_epochs):

    # Training loop per epoch
    for i, conformation in tqdm(enumerate(training), total=total_conformations, desc=f'epoch {lr_scheduler.last_epoch}'):
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
    print('RMSE:', validation_rmse, 'at epoch', lr_scheduler.last_epoch + 1)
    
    # Log per epoch info
    tensorboard.add_scalar('validation_rmse', validation_rmse, lr_scheduler.last_epoch)
    tensorboard.add_scalar('best_validation_rmse', lr_scheduler.best, lr_scheduler.last_epoch)
    tensorboard.add_scalar('learning_rate', optimizer.param_groups[0]['lr'] , lr_scheduler.last_epoch)
    
    # Early stopping
    if optimizer.param_groups[0]['lr'] < early_stopping_learning_rate:
        break

    # Checkpoint
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }, latest_checkpoint.as_posix())

    if lr_scheduler.is_better(validation_rmse, lr_scheduler.best):
        torch.save(model.state_dict(), best_model_checkpoint.as_posix())
    
    # Step the scheduler (every epoch)
    lr_scheduler.step(validation_rmse)

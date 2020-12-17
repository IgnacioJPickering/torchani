import torch

class RootAtomsLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        return (self.mse(predicted, target) / num_atoms.sqrt()).mean()

class BareLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

    def forward(self, predicted, target, species):
        return (self.mse(predicted, target)).mean()

class MultiTaskLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=10):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            self.register_buffer('weights', torch.ones(num_inputs,
                dtype=torch.double) * 1 / num_inputs)

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
        squares = self.mse(predicted, target)
        losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskRelativeLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=10):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            self.register_buffer('weights', torch.ones(num_inputs,
                dtype=torch.double) * 1 / num_inputs)

    def forward(self, predicted, target, species):
        square_ground = self.mse(predicted[:, 0], target[:, 0])
        ratios_ex = predicted[:, 1:]/target[:, 1:]
        squares_ex = self.mse(ratios_ex, torch.ones_like(ratios_ex))
        squares = torch.cat((square_ground.unsqueeze(-1), squares_ex), dim=-1)
        losses = (squares).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskSpectraLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11, num_other_inputs=10, dipoles=False):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            if dipoles:
                # I will rescale the dipoles manually with weights if needed
                self.register_buffer('weights', torch.ones(num_inputs + 3 * num_other_inputs,
                    dtype=torch.double) * 1 / (num_inputs + 3 * num_other_inputs))
            else:
                self.register_buffer('weights', torch.ones(num_inputs + num_other_inputs,
                    dtype=torch.double) * 1 / (num_inputs + num_other_inputs))

        self.num_other_inputs = num_other_inputs
        self.dipoles = dipoles

    def forward(self, predicted, target,  target_other, predicted_other, species):

        target = torch.cat((target, target_other), dim=-1)
        predicted = torch.cat((predicted, predicted_other), dim=-1)
        losses = (self.mse(predicted, target)).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskBareLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            self.register_buffer('weights', torch.ones(num_inputs,
                dtype=torch.double) * 1 / num_inputs)

    def forward(self, predicted, target, species):
        losses = self.mse(predicted, target).mean(dim=0)
        loss = (losses * self.weights).sum()
        return  loss, losses.detach()

class MultiTaskPairwiseLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        if weights is not None:
            self.register_buffer('weights', torch.tensor(weights,
                dtype=torch.double))
        else:
            pairs = num_inputs * (num_inputs - 1) // 2
            self.register_buffer('weights', torch.ones(pairs,
                dtype=torch.double) * 1 / pairs)

        row_major = torch.arange(0, num_inputs* num_inputs).reshape(num_inputs, num_inputs)
        idxs = torch.triu(row_major, diagonal=1)
        idxs = torch.masked_select(idxs, idxs != 0)
        self.register_buffer('idxs', idxs)

    def forward(self, predicted, target, species):
        num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)

        # get absolute energies
        predicted[1:] = predicted[1:] + predicted[0]
        target[1:] = target[1:] + target[0]

        # get all pairwise differences
        diff = target.unsqueeze(2) - target.unsqueeze(1)
        diff_predicted = predicted.unsqueeze(2) - predicted.unsqueeze(1)

        diff_target = diff.flatten(start_dim=1)[:, self.idxs]
        diff_predicted = diff_predicted.flatten(start_dim=1)[:, self.idxs]

        squares = self.mse(diff_predicted, diff_target)
        losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        loss = (losses * self.weights).sum()
        # I don't need one million losses so I don't even output them
        return  loss, None

class MTLLoss(torch.nn.Module):
    """A class to calculate the MTL loss with homoscedastic uncertainty
    https://arxiv.org/abs/1705.07115
    Args:
        losses: a list of task specific loss terms
        num_tasks: number of tasks
    """
    def __init__(self, num_tasks=11):
        super(MTLLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_sigma = torch.nn.Parameter(torch.zeros((num_tasks)))
    def get_precisions(self):
        return 0.5 * torch.exp(- self.log_sigma) ** 2

    def forward(self, *loss_terms):
        assert len(loss_terms) == self.num_tasks
        total_loss = 0
        self.precisions = self.get_precisions()
        for task in range(self.num_tasks):
            total_loss += self.precisions[task] * loss_terms[task] + self.log_sigma[task]
        return total_loss

class MtlSpectraLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11, num_other_inputs=10, dipoles=False):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

        self.num_other_inputs = num_other_inputs
        self.dipoles = dipoles
        self.num_tasks = num_inputs + num_other_inputs

        self.log_sigma = torch.nn.Parameter(torch.zeros((num_inputs + num_other_inputs)))

    def get_precisions(self):
        return 0.5 * torch.exp(- self.log_sigma) ** 2

    def forward(self, predicted, target,  target_other, predicted_other, species):

        target = torch.cat((target, target_other), dim=-1)
        predicted = torch.cat((predicted, predicted_other), dim=-1)
        assert len(target[1]) == self.num_tasks
        self.precisions = self.get_precisions()

        losses = (self.mse(predicted, target)).mean(dim=0)
        loss = (losses * self.precisions).sum() + self.log_sigma.sum()
        return  loss, losses.detach()

class MtlThreeSpectraLoss(torch.nn.Module):
    # this function can be used even if the input has multiple
    # values, in that case it just adds up the values, multiplies 
    # them by weights (or performs an average) and outputs both the individual
    # values and the sum as a loss

    def __init__(self, weights=None, num_inputs=11, num_other_inputs=10, dipoles=False):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')

        self.num_other_inputs = num_other_inputs
        self.dipoles = dipoles
        self.num_inputs = num_inputs
        self.num_tasks = num_inputs + num_other_inputs
        self.log_sigma = torch.nn.Parameter(torch.zeros(3))

    def get_precisions(self):
        return 0.5 * torch.exp(- self.log_sigma) ** 2

    def forward(self, predicted, target,  target_other, predicted_other, species):

        target = torch.cat((target, target_other), dim=-1)
        predicted = torch.cat((predicted, predicted_other), dim=-1)
        self.precisions = self.get_precisions()

        losses = (self.mse(predicted, target)).mean(dim=0)
        loss = 0.0
        loss += losses[0] * self.precisions[0]
        loss += (losses[1:self.num_inputs] * self.precisions[1]).sum()
        loss += (losses[self.num_inputs:self.num_tasks] * self.precisions[2]).sum()
        loss += self.log_sigma.sum()

        return  loss, losses.detach()

class MultiTaskUncertaintyLoss(torch.nn.Module):

    def __init__(self, num_inputs=10, weight_sqrt_atoms=True):
        super().__init__()
        self.mse =  torch.nn.MSELoss(reduction='none')
        self.weight_sqrt_atoms = weight_sqrt_atoms
        # predict log_sigmas_squared since it is more numerically stable
        # this is equivalent to initializing sigmas as ones
        self.register_parameter('log_sigmas_squared', torch.nn.Parameter(torch.zeros(num_inputs, dtype=torch.double)))

    def forward(self, predicted, target, species):
        squares = self.mse(predicted, target) 
        if self.weight_sqrt_atoms:
            num_atoms = (species >= 0).sum(dim=1, dtype=target.dtype)
            losses = (squares / num_atoms.sqrt().reshape(-1, 1)).mean(dim=0)
        else:
            losses = (squares).mean(dim=0)
        loss = (0.5 * torch.exp(-self.log_sigmas_squared) * losses).sum() 
        loss = loss + 0.5 * self.log_sigmas_squared.sum()
        return  loss, losses.detach()


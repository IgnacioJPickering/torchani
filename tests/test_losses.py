import torch
import unittest
from torchani import training


class TestLosses(unittest.TestCase):
    # Test that checks that the friendly constructor
    # reproduces the values from ANI1x with the correct parameters
    def testRootAtom(self):
        target = torch.ones(10, dtype=torch.float) 
        predicted = torch.ones(10, dtype=torch.float) * 2
        species = torch.ones(40, dtype=torch.float).reshape(10, -1)
        loss_function = training.RootAtomsLoss()
        loss = loss_function(predicted, target, species)
        self.assertTrue(loss.item() == 0.5)

    def testMultiTask(self):
        target = torch.ones(10,5, dtype=torch.float) 
        predicted = torch.ones(10,5, dtype=torch.float) * 2
        predicted.requires_grad_(True)
        species = torch.ones(40, dtype=torch.float).reshape(10, -1)
        loss_function = training.MultiTaskLoss(num_inputs=target.shape[-1])
        loss, losses = loss_function(predicted, target, species)
        self.assertTrue(loss.item() == 0.5)
        self.assertTrue(losses.mean().item() == 0.5)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(losses.requires_grad)

    def testMultiTaskUncertainty(self):
        target = torch.ones(10,5, dtype=torch.float) 
        predicted = torch.ones(10,5, dtype=torch.float) * 2
        predicted.requires_grad_(True)
        species = torch.ones(40, dtype=torch.float).reshape(10, -1)
        loss_function = training.MultiTaskUncertaintyLoss(num_inputs=target.shape[-1])
        loss, losses = loss_function(predicted, target, species)
        self.assertTrue(loss_function.log_sigmas_squared[0].item() == 0)
        self.assertTrue(loss.item() == 1.250)
        self.assertTrue(losses.mean().item() == 0.5)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(losses.requires_grad)

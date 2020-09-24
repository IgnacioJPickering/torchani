import torch
import unittest
from collections import OrderedDict
from torchani.modules import AEVComputerSplit,ANIModelMultiple
from torchani.modules import AtomicNetworkClassic, AtomicNetworkResidual
from torchani.modules import AtomicNetworkResidualMultiple, AtomicNetworkSpecFlexMultiple
from torchani import AEVComputer, ANIModel
from torchani.training import reproducible_init_nobias, reproducible_init_bias

dims = {'dim_in': 384, 'dim_out1' : 192, 'dim_out2' : 96}


class TestVariantsAEV(unittest.TestCase):
    # Test that checks that the friendly constructor
    # reproduces the values from ANI1x with the correct parameters
    def testSplitter(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aev_split = AEVComputerSplit.like_ani1x().to(device)
        aev_computer = AEVComputer.like_ani1x().to(device)
        
        atoms = 10
        conformations = 1
        coordinates = torch.ones((conformations, atoms, 3), dtype=torch.float, device=device)
        species = torch.zeros((conformations, atoms), dtype=torch.long, device=device)
        
        species, aev = aev_computer((species, coordinates))
        species, radial, angular = aev_split((species, coordinates))
        
        radial = radial.reshape(conformations, atoms, -1)
        angular = angular.reshape(conformations, atoms, -1)
        cat_aev = torch.cat([radial, angular], dim=-1)
        self.assertTrue(torch.isclose(cat_aev, aev).all().item())


class TestVariantANIModels(unittest.TestCase):
    # test different variants of ani models
    def testMultiple(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        species = ['H', 'C', 'N', 'O']
        networks = OrderedDict([(s, AtomicNetworkClassic.like_ani1x('H')) for s in species])
        networks_reference = OrderedDict([(s, AtomicNetworkClassic.like_ani1x('H')) for s in species])
        model = ANIModelMultiple(networks, number_outputs=1, squeeze_last=True).to(device)
        model_reference = ANIModel(networks_reference).to(device)
        model.apply(reproducible_init_bias)
        model_reference.apply(reproducible_init_bias)
        species = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        aevs = torch.zeros((1, 4, 384), dtype=torch.float, device=device)
        self.assertTrue(torch.isclose(model((species, aevs)).energies, model_reference((species, aevs)).energies))


class TestAtomicNetworks(unittest.TestCase):

    def testClassic(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        an = AtomicNetworkClassic.like_ani1x('H').to(device)
        an.apply(reproducible_init_nobias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor(0., device=device)))
        an.apply(reproducible_init_bias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor(19.7846, device=device)))

    def testResidual(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        an = AtomicNetworkResidual(**dims).to(device)
        an.apply(reproducible_init_nobias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor(0., device=device)))
        an.apply(reproducible_init_bias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor(13696.0898, device=device)))


    def testResidualMultiple(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        an = AtomicNetworkResidualMultiple(**dims, number_outputs=5).to(device)
        an.apply(reproducible_init_nobias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([0.]*5, device=device)).all())
        an.apply(reproducible_init_bias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([13696.0898]*5, device=device)).all())

    def testResidualMultiple2(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = 5
        dims_shared = [192, 96]
        an = AtomicNetworkSpecFlexMultiple(dim_in= 384, dims_shared= list(dims.values())[1:], dims_specific=[], number_outputs=outputs).to(device)
        an.apply(reproducible_init_nobias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([0.]*outputs, device=device)).all())
        an.apply(reproducible_init_bias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([13696.0898]*outputs, device=device)).all())

        dims_specific = [96, 96]
        # another test, but now with some specific layers
        an = AtomicNetworkSpecFlexMultiple(dim_in= 384, dims_shared= dims_shared, dims_specific=dims_specific, number_outputs=5).to(device)
        an.apply(reproducible_init_nobias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([0.]*outputs, device=device)).all())
        an.apply(reproducible_init_bias)
        e = an(torch.zeros((1, 384), dtype=torch.float, device=device)).squeeze()
        self.assertTrue(torch.isclose(e, torch.tensor([1.187634085888e+12]*outputs, device=device)).all() )


    


if __name__ == '__main__':
    pass
    

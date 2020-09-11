"""Contains different versions of the main ANIModel module"""
from typing import Tuple, Optional
import torch
from torch import Tensor
from torchani.nn import SpeciesEnergies

class ANIModelMultiple(torch.nn.ModuleList):
    """ANI model that compute energies from species and AEVs.

    This version of ANIModel allows for outputs that have more than one
    dimension, such as predicting many excited states per each atom Everything
    is essentially the same, except that there is a new dimension, number
    ouptuts, that determines the number of parameters that are output per atom
    """
    def __init__(self, modules, number_outputs=1):
        super().__init__(modules)
        self.number_outputs=number_outputs
    
    def forward(
            self,
            species_aev: Tuple[Tensor, Tensor],
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        assert cell is None
        assert pbc is None
        #shape of species is now C, A
        species, aev = species_aev
        #shape of species is now C x A
        species_ = species.flatten()
        #shape of AEV is C x A, other
        aev = aev.flatten(0, 1)
        # shape of output will be (C x A, O)
        output = aev.new_zeros((species_.shape[0], self.number_outputs))
        for i, m in enumerate(self):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                mask = mask.unsqueeze(-1).repeat(1,self.number_outputs)
                output.masked_scatter_(mask, m(input_))
        output = output.reshape((*species.shape, -1))
        return SpeciesEnergies(species, torch.sum(output, dim=1))

class ANIModelShare(torch.nn.Module):
    """This version of ANIModel can share some parameters between all different
    atom types the shared parameters are passed as the first argument, and the
    distinct modules are passed as the second parameter, the shared model is
    only applied once, to the full aev, and the distinct models are applied
    afterwards

    I don't think this is actually a specially good idea, it is probably better
    to directly modify BuiltinModel, or wrap stuff inside Sequential
    """
    
    def __init__(self, shared_model, distinct_models):
        super().__init__()
        self.shared_model = shared_model
        self.distinct_models = torch.nn.ModuleList(distinct_models)
    
    def forward(
            self,
            species_aev: Tuple[Tensor, Tensor],
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        assert cell is None
        assert pbc is None
        species, aev = species_aev
        species_ = species.flatten()
        # flattens aev to be of dim C x A, aev_dim instead of (C, A, aev_dim)
        aev = aev.flatten(0, 1)
        output = aev.new_zeros(species_.shape)
        # first act on aev via shared model
        aev = self.shared_model(aev)
        
        for i, m in enumerate(self.distinct_models):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output = torch.masked_scatter(mask, m(input_).flatten())
        # multiply by a small number to lower loss
        output = output.view_as(species)
        return SpeciesEnergies(species, torch.sum(output, dim=1))

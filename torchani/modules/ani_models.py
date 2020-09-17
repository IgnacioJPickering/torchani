"""Contains different versions of the main ANIModel module"""
from typing import Tuple, Optional
import torch
from torch import Tensor
from ..nn import ANIModel, SpeciesEnergies

class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1)

class ANIModelMultiple(ANIModel):
    """ANI model that compute energies from species and AEVs.

    This version of ANIModel allows for outputs that have more than one
    dimension, such as predicting many excited states per each atom Everything
    is essentially the same, except that there is a new dimension, number
    ouptuts, that determines the number of parameters that are output per atom

    This version of ANIModel can share some parameters between all different
    atom types the shared parameters are passed as the first argument, and the
    distinct modules are passed as the second parameter, the shared model is
    only applied once, to the full aev, and the distinct models are applied
    afterwards

    I don't think this is actually a specially good idea, it is probably better
    to directly modify BuiltinModel, or wrap stuff inside Sequential
    """

    def __init__(self, modules, shared_module=None, number_outputs=1, squeeze_last=False):
        # output can get squeezed if necessary for compatibility with normal
        # torchani

        super().__init__(modules)
        if shared_module is not None:
            self.shared_module = shared_module
        else:
            self.shared_module = torch.nn.Identity()

        if squeeze_last and number_outputs == 1:
            self.squeezer = Squeezer()
        else:
            self.squeezer = torch.nn.Identity()

        self.number_outputs=number_outputs

    def forward(
            self,
            species_aev: Tuple[Tensor, Tensor],
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        #shape of species is now C, A
        #shape of species is now C x A
        species_ = species.flatten()
        #shape of AEV is C x A, other
        aev = aev.flatten(0, 1)
        # shape of output will be (C x A, O)
        output = aev.new_zeros((species_.shape[0], self.number_outputs))
    
        # optionally a pass through a shared model is performed
        aev = self.shared_module(aev)
        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                mask = mask.unsqueeze(-1).repeat(1,self.number_outputs)
                output.masked_scatter_(mask, m(input_))
        output = output.reshape((*species.shape, -1))
        output = self.squeezer(output)
        return SpeciesEnergies(species, torch.sum(output, dim=1))

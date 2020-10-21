"""Contains different versions of the main ANIModel module"""
from typing import Tuple, Optional, NamedTuple
import torch
from collections import OrderedDict
from torch import Tensor
from ..nn import ANIModel, SpeciesEnergies
from .atomic_networks import AtomicNetworkClassic

class Squeezer(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1)

class SpeciesEnergiesExtra(NamedTuple):
    # extra can be magnitudes (either dipole squared or fosc) or dipoles
    species: Tensor
    energies: Tensor
    extra: Tensor

class ANIModelMultiple(ANIModel):
    """ANI model that compute energies from species and AEVs.

    This version of ANIModel allows for outputs that have more than one
    dimension, such as predicting many excited states per each atom
    Everything is essentially the same, except that there is a new dimension,
    number ouptuts, that determines the number of parameters that are output
    per atom

    This version of ANIModel can share some parameters between all different
    atom types the shared parameters are passed as the first argument, and
    the distinct modules are passed as the second parameter, the shared model
    is only applied once, to the full aev, and the distinct models are
    applied afterwards

    I don't think this is actually a specially good idea, it is probably
    better to directly modify BuiltinModel, or wrap stuff inside Sequential
    """

    def __init__(self, modules, shared_module=None, num_outputs=1,
            squeeze_last=False):
        # output can get squeezed if necessary for compatibility 
        # with normal torchani

        super().__init__(modules)
        if shared_module is not None:
            self.shared_module = shared_module
        else:
            self.shared_module = torch.nn.Identity()

        if squeeze_last and num_outputs == 1:
            self.squeezer = Squeezer()
        else:
            self.squeezer = torch.nn.Identity()

        self.num_outputs=num_outputs

    @classmethod
    def like_ani1x(cls):
        species = ['H', 'C', 'N', 'O']
        distinct_list = [(s, 
            AtomicNetworkClassic.like_ani1x(s)) for s in species]
        return cls(OrderedDict(distinct_list), squeeze_last=True)

    @classmethod
    def like_ani1ccx(cls):
        # just a synonym
        return cls.like_ani1x()

    @classmethod
    def like_ani2x(cls):
        species = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        distinct_list = [(s, 
            AtomicNetworkClassic.like_ani2x(s)) for s in species]
        return cls(OrderedDict(distinct_list), squeeze_last=True)

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
        output = aev.new_zeros((species_.shape[0], self.num_outputs))

        # optionally a pass through a shared model is performed
        aev = self.shared_module(aev)
        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                mask = mask.unsqueeze(-1).repeat(1,self.num_outputs)
                output.masked_scatter_(mask, m(input_))

        output = output.reshape((*species.shape, -1))
        output = self.squeezer(output)
        return SpeciesEnergies(species, torch.sum(output, dim=1))

class ANIModelScaledMultiple(ANIModelMultiple):

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
        output = aev.new_zeros((species_.shape[0], self.num_outputs))

        # optionally a pass through a shared model is performed
        aev = self.shared_module(aev)
        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                mask = mask.unsqueeze(-1).repeat(1,self.num_outputs)
                output.masked_scatter_(mask, m(input_))

        output = output.reshape((*species.shape, -1))
        output = self.squeezer(output)
        
        # normalize output of excited states because
        # excited state energies are always order 1
        output[:, :, 1:] = output[:, :, 1:]/(species >= 0).sum(dim=1).reshape(-1, 1, 1)

        return SpeciesEnergies(species, torch.sum(output, dim=1))

class ANIModelDipoles(ANIModelMultiple):
    """ANI model that compute energies and charges, 
    can output dipoles if asked to do so"""
    def __init__(self, modules, shared_module=None, num_outputs=1,
            squeeze_last=False, dipoles=False):
        super().__init__(modules, shared_module, num_outputs, squeeze_last)
        self.register_buffer('calc_dipoles', torch.bool(dipoles))

    def forward(
            self,
            species_coordinates_aev: Tuple[Tensor, Tensor, Tensor],
            cell: Optional[Tensor] = None,
            pbc: Optional[Tensor] = None) -> SpeciesEnergiesExtra:
        species, coordinates, aev = species_coordinates_aev
        assert species.shape == aev.shape[:-1]
        #shape of species is now C, A
        #shape of species is now C x A
        species_ = species.flatten()
        #shape of AEV is C x A, other
        aev = aev.flatten(0, 1)
        # shape of output will be (C x A, O)
        output_energies = aev.new_zeros(
                (species_.shape[0], self.num_outputs))
        output_charges = aev.new_zeros(
                (species_.shape[0], self.num_outputs))
        # optionally a pass through a shared model is performed
        aev = self.shared_module(aev)
        for i, m in enumerate(self.values()):
            mask = (species_ == i)
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                mask = mask.unsqueeze(-1).repeat(1,self.num_outputs)
                # this has to be coupled with a network that can handle
                # charges somehow
                energies, charges = m(input_)
                output_energies.masked_scatter_(mask, energies)
                output_charges.masked_scatter_(mask, charges)
        output_energies = output_energies.reshape((*species.shape, -1))
        output_charges = output_charges.reshape((*species.shape, -1))
        output_energies = self.squeezer(output_energies)
        output_charges = self.squeezer(output_charges) 
        # shape of charges is (C, A, X)
        total_energies = output_energies.sum(dim=1)
        # unsqueezed output charges have shape (C, A, 1, X)
        # unsqueezed coordinates are (C, A, 3, 1)
        if self.calc_dipoles.item():
            extra = (coordinates.unsqueeze(3) *\
                    output_charges.unsqueeze(2)).sum(dim=1)
        else:
            extra = output_charges.sum(dim=1)
        # total dipoles are (C, 3, X)
        # since we are outputting charges output of each network will be
        # tuple energies, charges
        return SpeciesEnergiesExtra(species, total_energies, extra)

from ..nn import SpeciesEnergies, SpeciesConverter
from ..aev import AEVComputer
from .shifter import EnergyShifter
from .ani_models import ANIModelMultiple
import torch
from torch import Tensor
from typing import Optional, Tuple

class TemplateModel(torch.nn.Module):
    r"""Template for ANI models"""

    def __init__(self, species_converter, aev_computer, neural_networks,
            energy_shifter, periodic_table_index=True, shift=True):
        super().__init__()

        # internal modules
        self.species_converter = species_converter
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter

        self.register_buffer('periodic_table_index',
                torch.tensor(periodic_table_index, dtype=torch.bool))
        self.register_buffer('shift_before_output', torch.tensor(shift, dtype=torch.bool))

    @classmethod
    def like_ani1x(cls, shift=True, periodic_table_index=True):
        kwargs = {
                'species_converter' : SpeciesConverter.like_ani1x(), 
                'energy_shifter' : EnergyShifter.like_ani1x(), 
                'aev_computer' : AEVComputer.like_ani1x(), 
                'neural_networks' : ANIModelMultiple.like_ani1x(), 
                'shift' : shift, 
                'periodic_table_index': periodic_table_index
        }
        return cls(**kwargs)

    @classmethod
    def like_ani1ccx(cls, shift=True, periodic_table_index=True):
        kwargs = {
                'species_converter' : SpeciesConverter.like_ani1ccx(), 
                'energy_shifter' : EnergyShifter.like_ani1ccx(), 
                'aev_computer' : AEVComputer.like_ani1ccx(), 
                'neural_networks' : ANIModelMultiple.like_ani1ccx(), 
                'shift' : shift, 
                'periodic_table_index': periodic_table_index
        }
        return cls(**kwargs)

    @classmethod
    def like_ani2x(cls, shift=True, periodic_table_index=True):
        kwargs = {
                'species_converter' : SpeciesConverter.like_ani2x(), 
                'energy_shifter' : EnergyShifter.like_ani2x(), 
                'aev_computer' : AEVComputer.like_ani2x(), 
                'neural_networks' : ANIModelMultiple.like_ani2x(), 
                'shift' : shift, 
                'periodic_table_index': periodic_table_index
        }
        return cls(**kwargs)

    def species_order(self):
        # get the order from the species converter
        return self.species_converter.species_order()
    
    def shift_before_output_(self, shift):
        assert isinstance(shift, bool)
        self.register_parameter('shift_before_output', torch.tensor(shift, dtype=torch.bool))
        return self

    def periodic_table_index_(self, use):
        assert isinstance(use, bool)
        self.register_parameter('periodic_table_index', torch.tensor(use, dtype=torch.bool))
        return self

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)

        # check if unknown species are included
        if species_coordinates[0].ge(self.aev_computer.num_species).any():
            raise ValueError(f'Unknown species found in {species_coordinates[0]}')

        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)

        if self.shift_before_output:
            species_energies = self.energy_shifter(species_energies)
        return species_energies

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def ase(self, **kwargs):
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)

from collections import OrderedDict
from ..nn import SpeciesEnergies, SpeciesConverter
from ..aev import AEVComputer
from .shifter import EnergyShifter
from .ani_models import ANIModelMultiple
from torchani import modules
import torch
from torch import Tensor
from typing import Optional, Tuple
from pathlib import Path
import yaml

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
    def from_yaml(cls, yaml_file):
        # Create a template model from a Yaml file with specified architecture
        if isinstance(yaml_file, str):
            with open(Path(yaml_file).resolve(), 'r') as f:
                hyper = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(yaml_file, Path):
            with open(yaml_file.resolve(), 'r') as f:
                hyper = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(yaml_file, dict):
            keys = list(yaml_file.keys())
            assert 'species_converter' in keys
            assert 'aev_computer' in keys
            assert 'ani_model' in keys
            assert 'atomic_network' in keys
            assert 'energy_shifter' in keys
            hyper = yaml_file


        class_ani_model = getattr(modules, hyper['ani_model'].pop('class'))
        class_aev_computer = getattr(modules, hyper['aev_computer'].pop('class'))
        class_atomic_network = getattr(modules, hyper['atomic_network'].pop('class'))
        
        # this is used to ensure that all of the pieces fit together correctly
        species = hyper['species_converter']['species']
        assert len(species) == hyper['aev_computer']['num_species']
        assert len(species) == hyper['energy_shifter']['num_species']

        atomic_networks = OrderedDict([ (s, class_atomic_network(**hyper['atomic_network'])) for 
                s in species])
        kwargs = {
                'species_converter' : SpeciesConverter(**hyper['species_converter']), 
                'aev_computer' : class_aev_computer.cover_linearly(**hyper['aev_computer']), 
                'neural_networks' : class_ani_model(atomic_networks, **hyper['ani_model']), 
                'energy_shifter' : EnergyShifter(**hyper['energy_shifter'])
        }
        return cls(**kwargs)
        

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
        self.shift_before_output = torch.tensor(shift, dtype=torch.bool)
        return self

    def periodic_table_index_(self, use):
        assert isinstance(use, bool)
        self.periodic_table_index = torch.tensor(use, dtype=torch.bool)
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

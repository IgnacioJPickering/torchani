from torch import Tensor
from typing import Tuple, Optional, NamedTuple
import torch

from .aev_computer_joint import AEVComputerJoint, SpeciesAEV

class SpeciesSplitAEV(NamedTuple):
    species: Tensor
    radial: Tensor
    angular: Tensor

class SpeciesCoordinatesAEV(NamedTuple):
    species: Tensor
    coordinates: Tensor
    aevs: Tensor


class AEVComputerSplit(AEVComputerJoint):
    r"""AEV Computer that splits the final aev into correctly shaped radial and angular parts

    Useful for different refinement of angular and radial subsections of the AEV
    """
    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesSplitAEV:
        """Compute AEVs, return in correct shape
        """
        species, coordinates = input_
        assert species.shape == coordinates.shape[:-1]

        if cell is None and pbc is None:
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), None)
        else:
            assert (cell is not None and pbc is not None)
            cutoff = max(self.Rcr, self.Rca)
            shifts = self.compute_shifts(cell, pbc, cutoff)
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), (cell, shifts))
        radial_aev, angular_aev = torch.split(aev, [self.radial_length, self.angular_length], dim=-1)
        num_species_pairs = (((self.num_species + 1) * (self.num_species))//2)
        angular_dist_divisions = len(self.ShfA)
        radial_dist_divisions = len(self.ShfR)
        angle_sections = len(self.ShfZ)

        assert radial_aev.shape[-1] + angular_aev.shape[-1] == self.aev_length
        assert radial_aev.shape[-1]//self.num_species == self.radial_sublength
        assert angular_aev.shape[-1]//num_species_pairs == self.angular_sublength
        # here I split the AEV and resize it so that it has some logical size
        # shape
        # for ani1x this is -1, 4, 16 for radial and -1 10, 4, 8 for angular
        # first dimension is batch dimension
        radial_aev = radial_aev.reshape(-1, self.num_species, radial_dist_divisions) #(species, radial value)
        angular_aev = angular_aev.reshape(-1, num_species_pairs, angular_dist_divisions, angle_sections) #(species_pairs, radial_value, angular_value)

        return SpeciesSplitAEV(species, radial_aev, angular_aev)

class AEVComputerCoord(AEVComputerJoint):

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None,
                masses: Optional[Tensor] = None) -> SpeciesAEV:
        species, coordinates = input_
        assert species.shape == coordinates.shape[:-1]

        if cell is None and pbc is None:
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), None)
        else:
            assert (cell is not None and pbc is not None)
            shifts = self.compute_shifts(cell, pbc, self.cutoff)
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), (cell, shifts))

        # if masses is passed, shape (C, A) then these are used to calculate
        # the center of mass for each molecule and displace all coordinates

        if masses is not None:
            assert masses.shape == species.shape
            total_masses = masses.sum(dim=1) # masses are summed across all atoms
            com = (coordinates * masses.unsqueeze(-1)).sum(2)/ total_masses.unsqueeze(-1)
            assert com.shape[0] == coordinates.shape[0]
            assert com.shape[1] == 3
            coordinates = coordinates - com

        # coordinates get passed onto the ANIModel in order to calculate dipoles
        return SpeciesCoordinatesAEV(species, aev, coordinates)

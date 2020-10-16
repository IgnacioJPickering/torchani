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

class AEVComputerNorm(AEVComputerJoint):
    r"""This AEV computer uses functions that are properly scaled to 
    lie in a range 0 - 1 for both radial and angular coefficients"""

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ,
            num_species, angle_scales=None, radial_scales=None, 
            trainable_radial_shifts=False,
            trainable_angular_shifts=False,
            trainable_angle_sections=False,
            trainable_etas=False, trainable_zeta=False, trainable_shifts=False):
        super().__init__(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, 
                num_species, trainable_radial_shifts, trainable_angular_shifts, 
                trainable_angle_sections, trainable_etas, trainable_zeta, trainable_shifts)

        if angle_scales is None:
            angle_scales = torch.tensor([
            0.7274285152554174, 
            0.3704739662625683, 
            0.11249284672261844, 
            0.0144858545390771], dtype=torch.float)
        if radial_scales is None:
            radial_scales = torch.tensor([
            0.9282837113533762, 
            0.8810616615689186, 
            0.823926001674413, 
            0.7584146320997143, 
            0.68612829719817, 
            0.6089622857973057, 
            0.5290817141020113, 
            0.4484833462374985, 
            0.36930025215334017, 
            0.29361443294889067, 
            0.22345529875946749, 
            0.1606157294729754, 
            0.10676448215557516, 
            0.06328044533463194, 
            0.03121678259062526, 
            0.011104079116615408], dtype=torch.float)
        

        # angle scales has same shape as ShfA/EtaA
        self.register_buffer('angle_scales', angle_scales.view(-1, 1))
        self.register_buffer('radial_scales', radial_scales)

    def angular_terms(self, Rca: float, ShfZ: Tensor, EtaA: Tensor, Zeta: Tensor,
                      ShfA: Tensor, vectors12: Tensor) -> Tensor:
        vectors12 = vectors12.view(2, -1, 3, 1, 1)
        distances12 = vectors12.norm(2, dim=-3).sum(0) / 2
        
        # maybe this needs to be changed for numerical accuracy
        cos_angles = 0.95 * torch.nn.functional.cosine_similarity(vectors12[0], vectors12[1], dim=-3)
        angles = torch.acos(cos_angles)

        fc12 = self.cutoff_cosine(distances12, Rca) ** 2
        # add a normalization factor
        factor1 = (1 / 1.5 ** Zeta) * ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
        factor2 = (1 / self.angle_scales) * torch.exp(-EtaA * (distances12 - ShfA) ** 2)
        ret = factor1 * factor2 * fc12

        return ret.flatten(start_dim=1)

    def radial_terms(self, Rcr: float, EtaR: Tensor, ShfR: Tensor, distances: Tensor) -> Tensor:
        distances = distances.view(-1, 1)
        fc = self.cutoff_cosine(distances, Rcr)
        ret = (1 / self.radial_scales) * torch.exp(-EtaR * (distances - ShfR)**2) * fc
        return ret.flatten(start_dim=1)

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

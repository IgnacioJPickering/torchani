import torch
from torch import Tensor
from ..nn import SpeciesEnergies
from typing import Tuple, Optional


class EnergyShifter(torch.nn.Module):
    """Modified Energy Shifter class that is more reasonable"""
    def __init__(self, self_energies=None, intercept=0.0, num_species=1, num_outputs=1):
        super().__init__()

        if self_energies is None and num_outputs == 1:
            self_energies = torch.zeros(num_species, dtype=torch.double)
        elif self_energies is None and num_outputs > 0:
            self_energies = torch.zeros((num_species, num_outputs), dtype=torch.double)
        else:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        intercept = torch.tensor(intercept, dtype=torch.double)

        self.register_buffer('self_energies',
                             self_energies)  # this can be none
        self.register_buffer('intercept', intercept)
        self.register_buffer('dummy_self_energy',
                             torch.tensor(0.0, dtype=torch.double))
        self.register_buffer('dummy_species', torch.tensor(-1,
                                                           dtype=torch.long))

    @classmethod
    def like_ani1x(cls):
        # order H C N O
        self_energies = [
            -0.60095297999999996996, -38.08316124000000257865,
            -54.70775770000000193249, -75.19446356000000264430
        ]
        intercept = 0.0
        return cls(self_energies=self_energies, intercept=intercept)

    @classmethod
    def like_ani1ccx(cls):
        # order H C N O
        self_energies = [
            -0.59915013249195381295, -38.03750806057355760004,
            -54.67448347695332699914, -75.16043537275567132383
        ]
        intercept = 0.0
        return cls(self_energies=self_energies, intercept=intercept)

    @classmethod
    def like_ani2x(cls):
        # order H C N O S F Cl, unfortunately
        self_energies = [
            -0.59785839438271337620, -38.08933878049794685694,
            -54.71196829862106625342, -75.19106774742085974594,
            -398.15771253349248581799, -99.80348506781633943774,
            -460.16819394210267546441
        ]
        intercept = 0.0
        return cls(self_energies=self_energies, intercept=intercept)

    def forward(self,
                species_energies: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        # even if energies is of shape (C, X) this ends up working perfectly fine
        # the only issue in that case is that intercept should be an array
        # of the correct shape also, if used. 
        # the important point in any case is to have the correct (S, X)
        # shape for self_energies
        species, energies = species_energies
        self_energies = self.self_energies[species]
        # set to zero all self energies of the dummy atoms
        self_energies[species == self.dummy_species] = self.dummy_self_energy
        energies = self_energies.sum(dim=1) + self.intercept
        return SpeciesEnergies(species, energies)

    def extra_repr(self):
        return f'self_energies={self.self_energies.detach().cpu().numpy()}, intercept={self.intercept}'


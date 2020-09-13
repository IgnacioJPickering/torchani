from torch import Tensor
from typing import Tuple, Optional, NamedTuple
import sys
from torchani.aev import compute_aev, compute_shifts, AEVComputer
import torch

if sys.version_info[:2] < (3, 7):
    class FakeFinal:
        def __getitem__(self, x):
            return x
    Final = FakeFinal()
else:
    from torch.jit import Final

class SpeciesSplitAEV(NamedTuple):
    species: Tensor
    radial: Tensor
    angular: Tensor


class AEVComputerSplit(AEVComputer):
    r"""AEV Computer that splits the final aev into correctly shaped radial and angular parts
    """
    Rcr: Final[float]
    Rca: Final[float]
    num_species: Final[int]

    radial_sublength: Final[int]
    radial_length: Final[int]
    angular_sublength: Final[int]
    angular_length: Final[int]
    aev_length: Final[int]
    sizes: Final[Tuple[int, int, int, int, int]]

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesSplitAEV:
        """Compute AEVs, return in correct shape
        """
        species, coordinates = input_
        assert species.shape == coordinates.shape[:-1]

        if cell is None and pbc is None:
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, None)
        else:
            assert (cell is not None and pbc is not None)
            cutoff = max(self.Rcr, self.Rca)
            shifts = compute_shifts(cell, pbc, cutoff)
            aev = compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes, (cell, shifts))

        radial_aev, angular_aev = torch.split(aev, [self.radial_length, self.angular_length])
        assert len(radial_aev) + len(angular_aev) == self.aev_length
        assert len(radial_aev)//self.num_species == self.radial_sublength
        assert len(angular_aev)//(((self.num_species + 1) * (self.num_species))//2) == self.angular_sublength
        # here I split the AEV and resize it so that it has some logical size
        # shape
        radial_aev = radial_aev.reshape(-1, 4, 16) #(species, radial value)
        angular_aev = angular_aev.reshape(-1, 10, 4, 8) #(species_pairs, radial_value, angular_value)

        return SpeciesSplitAEV(species, radial_aev, angular_aev)

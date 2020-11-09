from torch import Tensor
from typing import NamedTuple

from .aev_computer_joint import AEVComputerJoint

class SpeciesSplitAEV(NamedTuple):
    species: Tensor
    radial: Tensor
    angular: Tensor

class SpeciesCoordinatesAEV(NamedTuple):
    species: Tensor
    coordinates: Tensor
    aevs: Tensor

class AEVComputerNeighborlist(AEVComputerJoint):

    def __init__(self):
        pass


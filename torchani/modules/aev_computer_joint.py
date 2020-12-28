import torch

from torch import Tensor
import math
from typing import Tuple, Optional, NamedTuple
import numpy as np

class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor

class AEVComputerJoint(torch.nn.Module):
    """AEV Computer that has all functions inside as one class"""

    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ,
            num_species,
            trainable_radial_shifts=False,
            trainable_angular_shifts=False,
            trainable_angle_sections=False,
            trainable_etas=False, trainable_zeta=False, trainable_shifts=False
            ):
        super().__init__()
        self.register_buffer('Rcr', torch.tensor(Rcr, dtype=torch.double))
        self.register_buffer('Rca', torch.tensor(Rca, dtype=torch.double))
        assert Rca <= Rcr, "Current implementation of AEVComputer assumes Rca <= Rcr"
        self.register_buffer('num_species', torch.tensor(num_species, dtype=torch.long))
        if trainable_shifts:
            assert trainable_radial_shifts == False
            assert trainable_angular_shifts == False
            assert trainable_angle_sections == False
            trainable_radial_shifts=True
            trainable_angular_shifts=True
            trainable_angle_sections=True

        # convert constant tensors to a ready-to-broadcast shape
        # shape convension (EtaR/ShfR)
        # shape convension (..., ShfA/EtaA, ShfZ/Zeta)
        if trainable_etas:
            if len(EtaR) == 1:
                EtaR = EtaR.repeat(len(ShfR))
            if len(EtaA) == 1:
                EtaA = EtaA.repeat(len(ShfA))
            self.register_parameter('EtaR', torch.nn.Parameter(EtaR))
            self.register_parameter('EtaA', torch.nn.Parameter(EtaA.view(-1, 1)))
        else:
            self.register_buffer('EtaR', EtaR)
            self.register_buffer('EtaA', EtaA.view(-1, 1))
        if trainable_zeta:
            if len(Zeta) == 1:
                Zeta = Zeta.repeat(len(ShfZ))
            self.register_parameter('Zeta', torch.nn.Parameter(Zeta.view(1, -1)))
        else:
            self.register_buffer('Zeta', Zeta.view(1, -1))
        if trainable_radial_shifts:
            self.register_parameter('ShfR', torch.nn.Parameter(ShfR))
        else:
            self.register_buffer('ShfR', ShfR)
        if trainable_angular_shifts:
            self.register_parameter('ShfA', torch.nn.Parameter(ShfA.view(-1, 1)))
        else:
            self.register_buffer('ShfA', ShfA.view(-1, 1))
        if trainable_angle_sections:
            self.register_parameter('ShfZ', torch.nn.Parameter(ShfZ.view(1, -1)))
        else:
            self.register_buffer('ShfZ', ShfZ.view(1, -1))

        # The length of radial subaev of a single species
        radial_sublength = self.ShfR.numel()
        self.register_buffer('radial_sublength', torch.tensor(radial_sublength, dtype=torch.long))
        # The length of full radial aev
        radial_length = self.num_species * self.radial_sublength
        self.register_buffer('radial_length', torch.tensor(radial_length.item(), dtype=torch.long))
        # The length of angular subaev of a single species
        angular_sublength =  self.ShfA.numel() * self.ShfZ.numel()
        self.register_buffer('angular_sublength', torch.tensor(angular_sublength, dtype=torch.long))
        # The length of full angular aev
        angular_length = (self.num_species * (self.num_species + 1)) // 2 * self.angular_sublength
        self.register_buffer('angular_length', torch.tensor(angular_length.item(), dtype=torch.long))
        # The length of full aev
        aev_length = self.radial_length + self.angular_length
        self.register_buffer('aev_length', torch.tensor(aev_length.item(), dtype=torch.long))

        self.register_buffer('triu_index', self.get_triu_index(num_species).to(device=self.EtaR.device))

        # Set up default cell and compute default shifts.
        # These values are used when cell and pbc switch are not given.
        self.register_buffer('cutoff', torch.tensor(max(self.Rcr, self.Rca).item(), dtype=torch.double))
        default_cell = torch.eye(3, dtype=self.EtaR.dtype, device=self.EtaR.device)
        default_pbc = torch.zeros(3, dtype=torch.bool, device=self.EtaR.device)
        default_shifts = self.compute_shifts(default_cell, default_pbc, self.cutoff)
        self.register_buffer('default_cell', default_cell)
        self.register_buffer('default_shifts', default_shifts)

    @staticmethod
    def cumsum_from_zero(input_: Tensor) -> Tensor:
        cumsum = torch.zeros_like(input_)
        torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
        return cumsum
    
    @staticmethod
    def get_triu_index(num_species: int) -> Tensor:
        species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
        pair_index = torch.arange(species1.shape[0], dtype=torch.long)
        ret = torch.zeros(num_species, num_species, dtype=torch.long)
        ret[species1, species2] = pair_index
        ret[species2, species1] = pair_index
        return ret

    @staticmethod
    def compute_shifts(cell: Tensor, pbc: Tensor, cutoff: Tensor) -> Tensor:
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration
    
        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:
                tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.
    
        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inv_distances = reciprocal_cell.norm(2, -1)
        num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
        num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
        # num repeats is 1, 1, 1 for logical values of small cutoffs, smaller
        # than the unit cell size
        # r1, r2, r3 are the three [1], and o is [0]
        r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)
        # the cartesian prod is just [1, 1, 1], [1, 1, 0], etc
        # and the catted output is [[1, 1, 1], [1, 1, 0], ... etc]
        # so it is a tensor of shape 13 x 3
        return torch.cat([
            torch.cartesian_prod(-r1,  o,  o),
            torch.cartesian_prod(-r1, -r2, o),
            torch.cartesian_prod(o,   -r2, o),
            torch.cartesian_prod(r1,  -r2, o),
            torch.cartesian_prod(-r1,  r2, -r3),
            torch.cartesian_prod(o,    r2, -r3),
            torch.cartesian_prod(r1,   r2, -r3),
            torch.cartesian_prod(-r1,  o,  -r3),
            torch.cartesian_prod(o,    o,  -r3),
            torch.cartesian_prod(r1,   o,  -r3),
            torch.cartesian_prod(-r1, -r2, -r3),
            torch.cartesian_prod(o,   -r2, -r3),
            torch.cartesian_prod(r1,  -r2, -r3),
        ])

    def sizes(self) -> Tuple[int, int, int, int, int]:
        return self.num_species.item(), self.radial_sublength.item(), self.radial_length.item(), self.angular_sublength.item(), self.angular_length.item()
    
    @staticmethod
    def cutoff_cosine(distances: Tensor, cutoff: float) -> Tensor:
        # assuming all elements in distances are smaller than cutoff
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5

    def angular_terms(self, Rca: float, ShfZ: Tensor, EtaA: Tensor, Zeta: Tensor,
                      ShfA: Tensor, vectors12: Tensor) -> Tensor:
        """Compute the angular subAEV terms of the center atom given neighbor pairs.
    
        This correspond to equation (4) in the `ANI paper`_. This function just
        compute the terms. The sum in the equation is not computed.
        The input tensor have shape (conformations, atoms, N), where N
        is the number of neighbor atom pairs within the cutoff radius and
        output tensor should have shape
        (conformations, atoms, ``self.angular_sublength()``)
    
        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        vectors12 = vectors12.view(2, -1, 3, 1, 1)
        distances12 = vectors12.norm(2, dim=-3)
    
        cos_angles = vectors12.prod(0).sum(1) / torch.clamp(distances12.prod(0), min=1e-10)
        # 0.95 is multiplied to the cos values to prevent acos from returning NaN.
        angles = torch.acos(0.95 * cos_angles)
    
        fcj12 = self.cutoff_cosine(distances12, Rca)
        factor1 = ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
        # TODO: Use x * x to avoid torchscript bug 
        e = (distances12.sum(0) / 2 - ShfA)
        factor2 = torch.exp(-EtaA * e* e)
        ret = 2 * factor1 * factor2 * fcj12.prod(0)
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?, ?) where ? depend on constants.
        # We then should flat the last 4 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)

    def radial_terms(self, Rcr: float, EtaR: Tensor, ShfR: Tensor, distances: Tensor) -> Tensor:
        """Compute the radial subAEV terms of the center atom given neighbors
    
        This correspond to equation (3) in the `ANI paper`_. This function just
        compute the terms. The sum in the equation is not computed.
        The input tensor have shape (conformations, atoms, N), where ``N``
        is the number of neighbor atoms within the cutoff radius and output
        tensor should have shape
        (conformations, atoms, ``self.radial_sublength()``)
    
        .. _ANI paper:
            http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract
        """
        distances = distances.view(-1, 1)
        fc = self.cutoff_cosine(distances, Rcr)
        # Note that in the equation in the paper there is no 0.25
        # coefficient, but in NeuroChem there is such a coefficient.
        # We choose to be consistent with NeuroChem instead of the paper here.
        # TODO: use x*x to avoid torchscript bug
        ret = 0.25 * torch.exp(-EtaR * (distances - ShfR)*(distances - ShfR)) * fc
        # At this point, ret now has shape
        # (conformations x atoms, ?, ?) where ? depend on constants.
        # We then should flat the last 2 dimensions to view the subAEV as a two
        # dimensional tensor (onnx doesn't support negative indices in flatten)
        return ret.flatten(start_dim=1)

    def triple_by_molecule(self, atom_index12: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Input: indices for pairs of atoms that are close to each other.
        each pair only appear once, i.e. only one of the pairs (1, 2) and
        (2, 1) exists.
    
        Output: indices for all central atoms and it pairs of neighbors. For
        example, if input has pair (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), then the output would have
        central atom 0, 1, 2, 3, 4 and for cental atom 0, its pairs of neighbors
        are (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
        """
        # convert representation from pair to central-others
        ai1 = atom_index12.view(-1)
        sorted_ai1, rev_indices = ai1.sort()
    
        # sort and compute unique key
        out = torch.unique_consecutive(sorted_ai1, return_inverse=False, return_counts=True)
        counts = out[1]
        uniqued_central_atom_index = out[0]
    
        # compute central_atom_index
        # TODO: one avoids TorchScript bug
        one = torch.ones(1, dtype=torch.long, device=atom_index12.device)
        counts_less_one = counts - one
        pair_sizes = counts * counts_less_one // 2
        pair_indices = torch.repeat_interleave(pair_sizes)
        central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)
    
        # do local combinations within unique key, assuming sorted
        m = counts.max().item() if counts.numel() > 0 else 0
        n = pair_sizes.shape[0]
        intra_pair_indices = torch.tril_indices(m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
        mask = (torch.arange(intra_pair_indices.shape[2], device=ai1.device) < pair_sizes.unsqueeze(1)).flatten()
        sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
        sorted_local_index12 += self.cumsum_from_zero(counts).index_select(0, pair_indices)
    
        # unsort result from last part
        local_index12 = rev_indices[sorted_local_index12]
    
        # compute mapping between representation of central-other to pair
        n = atom_index12.shape[1]
        sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
        return central_atom_index, local_index12 % n, sign12

    @staticmethod
    def neighbor_pairs(padding_mask: Tensor, coordinates: Tensor, cell: Tensor,
                       shifts: Tensor, cutoff: float) -> Tuple[Tensor, Tensor]:
        """Compute pairs of atoms that are neighbors
    
        Arguments:
            padding_mask (:class:`torch.Tensor`): boolean tensor of shape
                (molecules, atoms) for padding mask. 1 == is padding.
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            cutoff (float): the cutoff inside which atoms are considered pairs
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        # shifts consists of a tensor of shape 13 x 3
        # that has [[1, 1, 1], [1, 1, 0], ... etc]
        coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
        cell = cell.detach()
    
        # Step 2: center cell
        # torch.triu_indices is faster than combinations
        num_atoms = padding_mask.shape[1]
        p12_center = torch.triu_indices(num_atoms, num_atoms, 1, device=cell.device)
        shifts_center = shifts.new_zeros((p12_center.shape[1], 3))
    
        # Step 3: cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        all_atoms = torch.arange(num_atoms, device=cell.device)
        prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
        shift_index = prod[0]
        p12 = prod[1:]
        shifts_outside = shifts.index_select(0, shift_index)
    
        # Step 4: combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        p12_all = torch.cat([p12_center, p12], dim=1)
        shift_values = shifts_all.to(cell.dtype) @ cell
    
        # step 5, compute distances, and find all pairs within cutoff
        num_mols = padding_mask.shape[0]
        selected_coordinates = coordinates.index_select(1, p12_all.view(-1)).view(num_mols, 2, -1, 3)
        distances = (selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shift_values).norm(2, -1)
        in_cutoff = (distances <= cutoff).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        molecule_index *= num_atoms
        atom_index12 = p12_all[:, pair_index]
        shifts = shifts_all.index_select(0, pair_index)
        return molecule_index + atom_index12, shifts

    @staticmethod
    def neighbor_pairs_nopbc(padding_mask: Tensor, coordinates: Tensor, cutoff: float) -> Tensor:
        """Compute pairs of atoms that are neighbors (doesn't use PBC)
    
        This function bypasses the calculation of shifts and duplication
        of atoms in order to make calculations faster
    
        Arguments:
            padding_mask (:class:`torch.Tensor`): boolean tensor of shape
                (molecules, atoms) for padding mask. 1 == is padding.
            coordinates (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cutoff (float): the cutoff inside which atoms are considered pairs
        """
        coordinates = coordinates.detach()
        current_device = coordinates.device
        num_atoms = padding_mask.shape[1]
        num_mols = padding_mask.shape[0]
        p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
        p12_all_flattened = p12_all.view(-1)
    
        pair_coordinates = coordinates.index_select(1, p12_all_flattened).view(num_mols, 2, -1, 3)
        distances = (pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...]).norm(2, -1)
        padding_mask = padding_mask.index_select(1, p12_all_flattened).view(num_mols, 2, -1).any(dim=1)
        distances.masked_fill_(padding_mask, math.inf)
        in_cutoff = (distances <= cutoff).nonzero()
        molecule_index, pair_index = in_cutoff.unbind(1)
        molecule_index *= num_atoms
        atom_index12 = p12_all[:, pair_index] + molecule_index
        return atom_index12
    
    def compute_aev(self, species: Tensor, coordinates: Tensor, triu_index: Tensor,
                    constants: Tuple[float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor],
                    sizes: Tuple[int, int, int, int, int], cell_shifts: Optional[Tuple[Tensor, Tensor]]) -> Tensor:
        Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
        num_species, radial_sublength, radial_length, angular_sublength, angular_length = sizes
        num_molecules = species.shape[0]
        num_atoms = species.shape[1]
        num_species_pairs = angular_length // angular_sublength
        coordinates_ = coordinates
        coordinates = coordinates_.flatten(0, 1)
    
        # PBC calculation is bypassed if there are no shifts
        if cell_shifts is None:
            atom_index12 = self.neighbor_pairs_nopbc(species == -1, coordinates_, Rcr)
            selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
            vec = selected_coordinates[0] - selected_coordinates[1]
        else:
            cell, shifts = cell_shifts
            atom_index12, shifts = self.neighbor_pairs(species == -1, coordinates_, cell, shifts, Rcr)
            shift_values = shifts.to(cell.dtype) @ cell
            selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
            vec = selected_coordinates[0] - selected_coordinates[1] + shift_values
    
        species = species.flatten()
        species12 = species[atom_index12]
    
        distances = vec.norm(2, -1)
    
        # compute radial aev
        radial_terms_ = self.radial_terms(Rcr, EtaR, ShfR, distances)
        radial_aev = radial_terms_.new_zeros((num_molecules * num_atoms * num_species, radial_sublength))
        index12 = atom_index12 * num_species + species12.flip(0)
        radial_aev.index_add_(0, index12[0], radial_terms_)
        radial_aev.index_add_(0, index12[1], radial_terms_)
        radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)
    
        # Rca is usually much smaller than Rcr, using neighbor list with cutoff=Rcr is a waste of resources
        # Now we will get a smaller neighbor list that only cares about atoms with distances <= Rca
        even_closer_indices = (distances <= Rca).nonzero().flatten()
        atom_index12 = atom_index12.index_select(1, even_closer_indices)
        species12 = species12.index_select(1, even_closer_indices)
        vec = vec.index_select(0, even_closer_indices)
    
        # compute angular aev
        central_atom_index, pair_index12, sign12 = self.triple_by_molecule(atom_index12)
        species12_small = species12[:, pair_index12]
        vec12 = vec.index_select(0, pair_index12.view(-1)).view(2, -1, 3) * sign12.unsqueeze(-1)
        species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])
        angular_terms_ = self.angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
        angular_aev = angular_terms_.new_zeros((num_molecules * num_atoms * num_species_pairs, angular_sublength))
        index = central_atom_index * num_species_pairs + triu_index[species12_[0], species12_[1]]
        angular_aev.index_add_(0, index, angular_terms_)
        angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
        return torch.cat([radial_aev, angular_aev], dim=-1)

    @classmethod
    def cover_linearly(cls, radial_cutoff: float, angular_cutoff: float,
                       radial_eta = None, angular_eta = None,
                       radial_dist_divisions=16, angular_dist_divisions=4,
                       zeta=32.0, angle_sections=8, num_species=4,
                       angular_start: float = 0.9, radial_start: float = 0.9,
                       logspace=False, sync_spacings=False, radial_sigma = None,
                       angular_sigma = None, unique_etas=True,
                       adapt_etas=False,
                       trainable_zeta=False, trainable_etas=False,
                       trainable_radial_shifts=False,
                       trainable_angular_shifts=False,
                       trainable_angle_sections=False,
                       trainable_shifts=False):
        r""" Provides a convenient way to linearly fill cutoffs

        This is a user friendly constructor that builds an
        :class:`torchani.AEVComputer` where the subdivisions along the the
        distance dimension for the angular and radial sub-AEVs, and the angle
        sections for the angular sub-AEV, are linearly covered with shifts. By
        default the distance shifts start at 0.9 Angstroms.

        Note that radial/angular eta determines the precision of the
        corresponding gaussians.

        To reproduce the ANI-1x AEV's the signature ``(5.2, 3.5, 16.0, 8.0, 16, 4, 32.0, 8, 4)``
        can be used.
        """
        if not adapt_etas:
            assert not (angular_eta is None) == (angular_sigma is None)
            assert not (radial_eta is None) == (radial_sigma is None)
        else:
            assert angular_eta is None
            assert radial_eta is None
            assert angular_sigma is None
            assert angular_sigma is None

        # This is intended to be self documenting code that explains the way
        # the AEV parameters for ANI1x were chosen. This is not necessarily the
        # best or most optimal way but it is a relatively sensible default.
        if sync_spacings:
                assert radial_dist_divisions > angular_dist_divisions, 'This is needed for sync_spacings'

        Rcr = radial_cutoff
        Rca = angular_cutoff

        # this is valid for both unique and non unique
        Zeta = torch.tensor([float(zeta)])
        if logspace:
            # logarithmically spaced divisions
            ShfR = torch.tensor(np.geomspace(radial_start, radial_cutoff,
                radial_dist_divisions, endpoint=True), dtype=torch.float)
            if sync_spacings:
                ShfA = ShfR.clone()[0:angular_dist_divisions]
                Rca = ShfA[-1].item() + 0.01
            else:
                ShfA = torch.tensor(np.geomspace(angular_start, angular_cutoff,
                    angular_dist_divisions, endpoint=True), dtype=torch.float)
        else:
            # linearly spaced divisions
            ShfR = torch.linspace(radial_start, radial_cutoff, radial_dist_divisions + 1)[:-1]
            if sync_spacings:
                ShfA = ShfR.clone()[0:angular_dist_divisions]
                Rca = ShfA[-1].item() + 0.01
            else:
                ShfA = torch.linspace(angular_start, angular_cutoff,
                        angular_dist_divisions + 1)[:-1]
        angle_start = math.pi / (2 * angle_sections)

        ShfZ = (torch.linspace(0, math.pi, angle_sections + 1) + angle_start)[:-1]
        if adapt_etas:
            unique_etas = False
            factor = 1.3
            radial_sigma =[(ShfR[1] - ShfR[0])*0.5*factor] + [(ShfR[j] -
                ShfR[j-1])*0.5*factor for j in range(1, len(ShfR))]

            angular_sigma =[(ShfA[1] - ShfA[0])*0.5*factor] + [(ShfA[j] -
                ShfA[j-1])*0.5*factor for j in range(1, len(ShfA))]

        if radial_sigma is not None:
            radial_eta = 1/(2 * np.asarray(radial_sigma) ** 2)
            radial_eta = radial_eta.tolist()
            if len(radial_eta) == 1:
                radial_eta = radial_eta[0]
        if angular_sigma is not None:
            angular_eta = 1/(2 * np.asarray(angular_sigma) ** 2)
            angular_eta = angular_eta.tolist()
            if len(angular_eta) == 1:
                angular_eta = angular_eta[0]

        if unique_etas:
            EtaR = torch.tensor([float(radial_eta)])
            EtaA = torch.tensor([float(angular_eta)])
        else:
            EtaR = torch.tensor(radial_eta).float()
            EtaA = torch.tensor(angular_eta).float()
            assert len(EtaR) == radial_dist_divisions
            assert len(EtaA) == angular_dist_divisions


        assert len(ShfA) == angular_dist_divisions
        assert len(ShfR) == radial_dist_divisions

        return cls(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species,
                trainable_zeta=trainable_zeta, trainable_etas=trainable_etas,
                trainable_radial_shifts=trainable_radial_shifts,
                trainable_angular_shifts=trainable_angular_shifts,
                trainable_angle_sections=trainable_angle_sections,
                trainable_shifts=trainable_shifts)

    @classmethod
    def like_ani1x(cls):
        kwargs = {'radial_cutoff' : 5.2,
                'angular_cutoff' : 3.5,
                'radial_eta' : 16.0,
                'angular_eta': 8.0,
                'radial_dist_divisions': 16,
                'angular_dist_divisions': 4,
                'zeta': 32.0,
                'angle_sections': 8,
                'num_species': 4,
                'angular_start': 0.9,
                'radial_start': 0.9}
        return cls.cover_linearly(**kwargs)

    @classmethod
    def like_ani2x(cls):
        kwargs = {'radial_cutoff' : 5.1,
                'angular_cutoff' : 3.5,
                'radial_eta' : 19.7,
                'angular_eta': 12.5,
                'radial_dist_divisions': 16,
                'angular_dist_divisions': 8,
                'zeta': 14.1,
                'angle_sections': 4,
                'num_species': 7,
                'angular_start': 0.8,
                'radial_start': 0.8}
        # note that there is a small difference of 1 digit in one decimal place
        # in the eight element of ShfR this element is 2.6812 using this method
        # and 2.6813 for the actual network, but this is not significant for
        # retraining purposes
        return cls.cover_linearly(**kwargs)

    @classmethod
    def like_ani1ccx(cls):
        # just a synonym
        return cls.like_ani1x()

    def extra_repr(self):
        return f'Rcr={self.Rcr}, Rca={self.Rca}, EtaR={self.EtaR}, ShfR={self.ShfR}, EtaA={self.EtaA}, ShfA={self.ShfA}, Zeta={self.Zeta}, ShfZ={self.ShfZ}, num_species={self.num_species}'

    def constants(self):
        return self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

    def map2central(self, cell, coordinates, pbc):
        inv_cell = torch.inverse(cell)
        coordinates_cell = torch.matmul(coordinates, inv_cell)
        coordinates_cell -= coordinates_cell.floor() * pbc
        coordinates_cell[coordinates_cell < 0.0] += 1.0
        coordinates_cell[coordinates_cell >= 1.0] -= 1.0
        assert (coordinates_cell >= 0.0).all()
        assert (coordinates_cell < 1.0).all()
        coordinates = torch.matmul(coordinates_cell, cell)
        assert not torch.isnan(coordinates_cell).any()
        assert not torch.isinf(coordinates_cell).any()
        return coordinates

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesAEV:
        """Compute AEVs

        Arguments:
            input_ (tuple): Can be one of the following two cases:

                If you don't care about periodic boundary conditions at all,
                then input can be a tuple of two tensors: species, coordinates.
                species must have shape ``(N, A)``, coordinates must have shape
                ``(N, A, 3)`` where ``N`` is the number of molecules in a batch,
                and ``A`` is the number of atoms.

                .. warning::

                    The species must be indexed in 0, 1, 2, 3, ..., not the element
                    index in periodic table. Check :class:`torchani.SpeciesConverter`
                    if you want periodic table indexing.

                .. note:: The coordinates, and cell are in Angstrom.

                If you want to apply periodic boundary conditions, then the input
                would be a tuple of two tensors (species, coordinates) and two keyword
                arguments `cell=...` , and `pbc=...` where species and coordinates are
                the same as described above, cell is a tensor of shape (3, 3) of the
                three vectors defining unit cell:

                .. code-block:: python

                    tensor([[x1, y1, z1],
                            [x2, y2, z2],
                            [x3, y3, z3]])

                and pbc is boolean vector of size 3 storing if pbc is enabled
                for that direction.

        Returns:
            NamedTuple: Species and AEVs. species are the species from the input
            unchanged, and AEVs is a tensor of shape ``(N, A, self.aev_length())``
        """
        species, coordinates = input_
        assert species.dim() == 2
        assert species.shape == coordinates.shape[:-1]
        assert coordinates.shape[-1] == 3

        assert not torch.isnan(coordinates).any()
        assert not torch.isinf(coordinates).any()

        if cell is None and pbc is None:
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), None)
        else:
            assert (cell is not None and pbc is not None)
            shifts = self.compute_shifts(cell, pbc, self.cutoff)
            coordinates = self.map2central(cell, coordinates, pbc)
            aev = self.compute_aev(species, coordinates, self.triu_index, self.constants(), self.sizes(), (cell, shifts))
        assert not torch.isnan(aev).any()
        return SpeciesAEV(species, aev)

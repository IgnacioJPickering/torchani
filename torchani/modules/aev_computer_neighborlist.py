from typing import Tuple
from collections import namedtuple

import torch
from torch.nn import functional
from torch import Tensor

from .aev_computer_joint import AEVComputerJoint

def stable_argsort(input_: Tensor):
    # argsort is NOT stable, it doesn't preserve order of equal elements
    # this means that it is not possible to use argsort again to recover a 
    # mapping that will preserve the original order of the elements.

    # hack to ensure stable sorting, works in pytorch 1.6.0 but may break in
    # the future, ideally stable sorting should be used here if the length of
    # the array is 2050 or more pytorch uses stable sort, so I pad the array
    # with very large numbers to go over that length if the array is small,
    # then sort and then unpad. There should not be much overhead
    if len(input_) <= 2050: 
        padded_input = functional.pad(input_, (0, 2050),'constant', 9000000000000)
        sorted_ = torch.argsort(padded_input)[:len(input_)]
    else:
        sorted_ = torch.argsort(input_)
    return sorted_

def stable_sort(input_: Tensor):
    # sort is NOT stable, it doesn't preserve order of equal elements
    # this means that it is not possible to use argsort again to recover a 
    # mapping that will preserve the original order of the elements.

    # hack to ensure stable sorting, works in pytorch 1.6.0 but may break in
    # the future, ideally stable sorting should be used here if the length of
    # the array is 2050 or more pytorch uses stable sort, so I pad the array
    # with very large numbers to go over that length if the array is small,
    # then sort and then unpad. There should not be much overhead
    if len(input_) <= 2050: 
        padded_input = functional.pad(input_, (0, 2050),'constant', 9000000000000)
        sorted_ = torch.sort(padded_input)
        IdxVal = namedtuple('IdxVal', 'indices values')
        sorted_ = IdxVal(indices=sorted_.indices[:len(input_)], values=sorted_.values[:len(input_)])
    else:
        sorted_ = torch.sort(input_)
    return sorted_


def cumsum_from_zero(input_: Tensor) -> Tensor:
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum

class CellListComputer(torch.nn.Module):

    def __init__(self, cutoff, buckets_per_cutoff=1):
        super().__init__()
        self.cell_diagonal = None
        self.scaling_for_flat_index = None
        # buckets_per_cutoff is also the number of buckets that is scanned in
        # each direction it determines how fine grained the grid is, with
        # respect to the cutoff. This is 2 for amber, but 1 is useful for debug
        self.buckets_per_cutoff = buckets_per_cutoff
        self.cutoff = cutoff
        # Here I get the vector index displacements for the neighbors of an
        # arbitrary vector index I think these are enough (this is different
        # from pmemd)
        index_disp_1d = torch.arange(-self.buckets_per_cutoff, 1)
        # I choose all the displacements except for the zero
        # displacement that does nothing, which is the last one
        self.vector_index_displacement = torch.cartesian_prod(
                               index_disp_1d, index_disp_1d, index_disp_1d)[:-1]
        # This is 26 for 2 buckets and 17 for 1 bucket 
        # This is necessary for the image - atom map and atom - image map
        self.num_neighbors = len(self.vector_index_displacement)
        # Get the length of a bucket in the bucket grid  shape 3, (Bx, By, Bz)
        # The length is cutoff/buckets_per_cutoff + epsilon
        self.bucket_length = self._get_bucket_length()

    def setup_cell_parameters(self,  cell: Tensor):
        current_device = cell.device
        # NOTE: if this is an NVT calculation this can be called only once
        # without issue

        # For now this will only be valid in the case where cell is diagonal
        # (orthogonal) in this case the first dimension has to be divided by
        # cell(1,1) the second dimension by cell(2,2) and the third dimension
        # by cell(3, 3). I will assume coordinates are already inside the cell
        # to start with 

        # 1) Update the cell diagonal
        self.cell_diagonal = torch.diagonal(cell.detach())
        
        # 2) Get max bucket index (Gx, Gy, Gz)
        # which give the size of the grid of buckets that fully covers the
        # whole volume of the unit cell U, given by "cell", and the number of
        # flat buckets (F) (equal to the total number of buckets, F
        # 
        # Gx, Gy, Gz is 1 + maximum index for \vb{g} Flat bucket indices are
        # indices for the buckets written in row major order (or equivalently
        # dictionary order), the number F = Gx*Gy*Gz

        # bucket_length = B, unit cell U_mu = B * 3 - epsilon this means I need
        # 3 buckets to cover completely the whole thing so max bucket index
        # should be 2
        self.shape_buckets_grid = torch.floor(
                self.cell_diagonal / self.bucket_length).to(torch.long) + 1
        self.total_buckets = self.shape_buckets_grid.prod()
        
        # 3) This is needed to scale and flatten last dimension of bucket indices
        # for row major this is (Gy * Gz, Gz, 1)
        self.scaling_for_flat_index = torch.tensor(
                     [self.shape_buckets_grid[1] * self.shape_buckets_grid[2],
                      self.shape_buckets_grid[1], 1],
                     dtype = torch.long, device = current_device)
        
        # 4) create the vector_index -> flat_index conversion tensor
        # it is not really necessary to perform circular padding, 
        # since we can index the array using negative indices!
        self.vector_idx_to_flat = torch.arange(0, self.total_buckets)
        self.vector_idx_to_flat = self.vector_idx_to_flat.reshape(
                                                self.shape_buckets_grid[0], 
                                                self.shape_buckets_grid[1],
                                                self.shape_buckets_grid[2])

        return self.shape_buckets_grid, self.total_buckets

    def _get_bucket_length(self, extra_space: float =0.00001):
        # Get the size (Bx, By, Bz) of the buckets in the grid
        # extra space by default is consistent with Amber
        # The spherical factor is different from 1 in the case of nonorthogonal
        # boxes and accounts for the "spherical protrusion", not sure exactly
        # what that is but I think it is related to the fact that the sphere of
        # radius "cutoff" around an atom needs some extra space in
        # nonorthogonal boxes.
        spherical_factor = torch.ones(3, dtype=torch.double)
        bucket_length = (spherical_factor * self.cutoff / self.buckets_per_cutoff) + extra_space
        return bucket_length

    def to_flat_index(self, x: Tensor) -> Tensor:
        # Converts a tensor with bucket indices in the last dimension to a tensor with
        # flat bucket indices in the last dimension
        # if your tensor is (N1, ..., Nd, 3) this transforms the tensor into
        # (N1, ..., Nd), which holds the flat bucket indices
        assert self.scaling_for_flat_index is not None,\
            "Scaling for flat index has not been computed"
        assert x.shape[-1] == 3
        return (x * self.scaling_for_flat_index).sum(-1)

    def expand_into_neighbors(self, x: Tensor) -> Tensor:
        # transforms a tensor of shape (... 3) with vector indices 
        # in the last dimension into a tensor of shape (..., Eta, 3)
        # where Eta is the number of neighboring buckets, indexed by n
        assert self.vector_index_displacement is not None,\
            "Displacement for neighbors has not been computed"
        assert x.shape[-1] == 3
        x = x.unsqueeze(-2) + self.vector_index_displacement

        # sanity check
        assert x.shape[-1] == 3
        assert x.shape[-2] == self.vector_index_displacement.shape[0]
        return x

    def fractionalize_coordinates(self, coordinates: Tensor) -> Tensor:
        # Scale coordinates to box size
        # I make all my coordinates relative to the box size this means for
        # instance that if the coordinate is 3.15 times the cell length, it is
        # turned into 3.15; if it is 0.15 times the cell length, it is turned
        # into 0.15, etc
        fractional_coordinates = coordinates / self.cell_diagonal.reshape(1, 1, -1)
        assert (fractional_coordinates < 1.).all(),\
            "Some coordinates are outside the box"
        return fractional_coordinates

    def fractional_to_vector_bucket_indices(self, fractional: Tensor) -> Tensor:
        # transforms a tensor of fractional coordinates (shape (..., 3))
        # into a tensor of vector bucket indices (same shape)
        # Since the number of indices to iterate over is a cartesian product of 3
        # vectors it will grow with L^3 (volume) I think it is intelligent to first
        # get all indices and then apply as needed I will call the buckets "main
        # bucket" and "neighboring buckets"
        assert self.shape_buckets_grid is not None,\
            "Max bucket index not computed"
        out = torch.round(fractional * (self.shape_buckets_grid - 1))
        out = out.reshape(1, 1, -1).to(torch.long)
        return out

    @staticmethod
    def get_imidx_converters(x: Tensor) -> Tuple[Tensor, Tensor]:
        # this are the "image indices", indices that sort atoms in the order of
        # the flattened bucket index.  Only occupied buckets are considered, so
        # if a bucket is unoccupied the index is not taken into account.  for
        # example if the atoms are distributed as:
        # / 1 9 8 / - / 3 2 4 / 7 / 
        # where the bars delimit flat buckets, then the assoc. image indices
        # are:
        # / 0 1 2 / - / 3 4 5 / 6 / 
        # atom indices can be reconstructed from the image indices, so the
        # pairlist can be built with image indices and then at the end calling
        # atom_indices_from_image_indices[pairlist] you convert to atom_indices

        # imidx_from_atidx returns tensors that convert image indices into atom
        # indices and viceversa
        imidx_from_atidx  = stable_argsort(x.squeeze())
        atidx_from_imidx = stable_argsort(imidx_from_atidx)
        return imidx_from_atidx, atidx_from_imidx

    @staticmethod
    def get_atoms_in_flat_bucket_counts(main_bucket_flat_index, total_buckets):
        # NOTE: check if bincount if fast. (bincount is only useful for 1D
        # inputs) count in flat bucket: 3 0 0 0 ... 2 0 0 0 ... 1 0 1 0 ...,
        # shape is total buckets F cumulative buckets count has the number of
        # atoms BEFORE a given bucket cumulative buckets count: 0 3 3 3 ... 3 5
        # 5 5 ... 5 6 6 7 ...
        main_bucket_flat_index = main_bucket_flat_index.squeeze()
        count_in_flat_bucket = torch.bincount(main_bucket_flat_index,
                                              minlength=total_buckets) 
        cumcount_in_flat_bucket = cumsum_from_zero(count_in_flat_bucket)

        # this is A*
        max_in_bucket = count_in_flat_bucket.max() 
        return count_in_flat_bucket, cumcount_in_flat_bucket, max_in_bucket

    @staticmethod
    def sort_along_row(x: Tensor, max_value, row_for_sorting=1) -> Tensor:
        # reorder padded pairs by ordering according to lower part instead of upper
        # based on https://discuss.pytorch.org/t/sorting-2d-tensor-by-pairs-not-columnwise/59465
        assert x.dim() == 2, "The input must have 2 dimensions"
        assert x.shape[0] == 2, "The inut must be shape (2, ?)"
        assert row_for_sorting == 1 or row_for_sorting == 0
        if row_for_sorting == 1:
            aug = x[0, :] + x[1, :] * max_value
        elif row_for_sorting == 0:
            aug = x[0, :] * max_value + x[1, :]
        x = x.index_select(1, stable_sort(aug).indices)
        return x

    def get_within_image_pairs(self, flat_bucket_count, max_in_bucket, flat_bucket_cumcount):
        # max_in_bucket = maximum number of atoms contained in any bucket
        current_device = flat_bucket_count.device

        # get all indices f that have pairs inside
        # these are A(w) and Ac(w), and withpairs_flat_index is actually f(w)
        withpairs_flat_index = (flat_bucket_count > 1).nonzero(as_tuple=True)
        withpairs_flat_bucket_count = flat_bucket_count[withpairs_flat_index]
        withpairs_flat_bucket_cumcount = flat_bucket_cumcount[withpairs_flat_index]

        padded_pairs = torch.triu_indices(max_in_bucket, max_in_bucket,
                                          offset = 1,
                                          device = current_device)

        padded_pairs = self.sort_along_row(padded_pairs, max_in_bucket, row_for_sorting=1)
        padded_pairs = padded_pairs + withpairs_flat_bucket_cumcount.reshape(-1, 1, 1)
        padded_pairs = padded_pairs.permute(1, 0, 2).reshape(2, -1)

        max_pairs_in_bucket = max_in_bucket * (max_in_bucket - 1) // 2
        mask = torch.arange(0, max_pairs_in_bucket, device = current_device)
        num_buckets_with_pairs = len(withpairs_flat_index[0])
        mask = mask.repeat(num_buckets_with_pairs, 1)
        withpairs_count_pairs = withpairs_flat_bucket_count * (withpairs_flat_bucket_count - 1) // 2
        mask = (mask < withpairs_count_pairs.reshape(-1, 1)).reshape(-1)

        upper = torch.masked_select(padded_pairs[0], mask)
        lower = torch.masked_select(padded_pairs[1], mask)
        within_image_pairs = torch.stack((upper, lower), dim=0)
        return within_image_pairs

    def get_lower_between_image_pairs(self, neighbor_flat_bucket_count, neighbor_flat_bucket_cumcount, max_in_bucket):
        # 3) now I need the LOWER part
        # this gives, for each atom, for each neighbor bucket, all the
        # unpadded, unshifted atom neighbors
        # this is basically broadcasted to the shape of fna
        # shape is 1 x A x eta x A*
        atoms = neighbor_flat_bucket_count.shape[1]
        padded_atom_neighbors = torch.arange(0, max_in_bucket)
        padded_atom_neighbors = padded_atom_neighbors.reshape(1, 1, 1, -1)
        padded_atom_neighbors = padded_atom_neighbors.repeat(1, atoms, self.num_neighbors, 1)

        # now I need to add A(f' < fna) shift the padded atom neighbors to get
        # image indices I need to check here that the cumcount is correct since
        # it was technically done with imidx so I need to check correctnes of
        # both counting schemes, but first I create the mask to unpad
        # and then I shift to the correct indices
        mask = (padded_atom_neighbors < neighbor_flat_bucket_count.unsqueeze(-1))
        padded_atom_neighbors += neighbor_flat_bucket_cumcount.unsqueeze(-1)
        # the mask should have the same shape as padded_atom_neighbors, and
        # now all that is left is to apply the mask in order to unpad
        assert padded_atom_neighbors.shape == mask.shape
        lower = torch.masked_select(padded_atom_neighbors, mask)
        return lower


class AEVComputerNL(AEVComputerJoint):
    
    def __init__(self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ,
            num_species,
            trainable_radial_shifts=False,
            trainable_angular_shifts=False,
            trainable_angle_sections=False,
            trainable_etas=False, trainable_zeta=False, trainable_shifts=False, 
            constant_volume=False):
        super().__init__(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, 
                trainable_radial_shifts, trainable_angular_shifts, trainable_angle_sections, 
                trainable_etas, trainable_zeta, trainable_shifts)
        self.clist = CellListComputer(Rcr)
        self.constant_volume = constant_volume

    def get_bucket_indices(self, fractional_coordinates):
        main_bucket_vector_index =\
                self.clist.fractional_to_vector_bucket_indices(fractional_coordinates)
        main_bucket_flat_index =\
                self.clist.to_flat_index(main_bucket_vector_index)
        return main_bucket_vector_index, main_bucket_flat_index

    def get_neighbor_bucket_indices(self, main_bucket_vector_index):
        # This is actually pure neighbors, so it doesn't have 
        # "the bucket itself" 
        # These are 
        # - g(a, n),  shape 1 x A x Eta x 3 
        # - f(a, n),  shape 1 x A x Eta
        # These give, for each atom, the flat index or the vector index of its
        # neighbor buckets (neighbor buckets indexed by n).
        neighbor_bucket_vector_indices = self.clist.expand_into_neighbors(main_bucket_vector_index) 

        atoms = neighbor_bucket_vector_indices.shape[1]
        neighbors = neighbor_bucket_vector_indices.shape[2]
        neighbor_bucket_flat_indices = neighbor_bucket_vector_indices.reshape(-1, 3).unbind(1)
        neighbor_bucket_flat_indices = self.clist.vector_idx_to_flat[neighbor_bucket_flat_indices]
        neighbor_bucket_flat_indices = neighbor_bucket_flat_indices.reshape(1, atoms, neighbors)
        return neighbor_bucket_vector_indices, neighbor_bucket_flat_indices
    
    def neighbor_pairs(self, padding_mask: Tensor, coordinates: Tensor, cell: Tensor,
                       shifts: Tensor, cutoff: float) -> Tuple[Tensor, Tensor]:
        # 1) Setup cell parameters, only once for constant V simulations, 
        # every time for variable V, (constant P) simulations
        if self.constant_volume and self.clist.total_buckets is None:
                shape_buckets_grid, total_buckets =\
                                        self.clist.setup_cell_parameters(cell)
        else:
            shape_buckets_grid, total_buckets =\
                                        self.clist.setup_cell_parameters(cell)

        # 2) Fractionalize coordinates
        fractional_coordinates = self.clist.fractionalize_coordinates(coordinates)
    
        # 3) Get vector indices and flattened indices for atoms in unit cell 
        # shape C x A x 3 this gives \vb{g}(a), the vector bucket idx
        # shape C x A this gives f(a), the flat bucket idx for atom a
        main_bucket_vector_index, main_bucket_flat_index =\
                                self.get_bucket_indices(fractional_coordinates)
        
        # 4) get image_indices -> atom_indices and inverse mapping
        # NOTE: there is not necessarily a requirement to do this here
        # both shape A, a(i) and i(a)
        imidx_from_atidx, atidx_from_imidx = self.clist.get_imidx_converters(
                                                        main_bucket_flat_index)
    
        # FIRST WE WANT "WITHIN" IMAGE PAIRS
        # 1) Get the number of atoms in each bucket (as indexed with f idx)
        # this gives A*, A(f) , "A(f' <= f)" = Ac(f) (cumulative) f being the
        # flat bucket index, A being the number of atoms for that bucket, 
        # and Ac being the cumulative number of atoms up to that bucket
        flat_bucket_count, flat_bucket_cumcount, max_in_bucket =\
        self.clist.get_atoms_in_flat_bucket_counts(main_bucket_flat_index,
                total_buckets)
    
        # 2) this are indices WITHIN the central buckets
        within_image_pairs =\
        self.clist.get_within_image_pairs(flat_bucket_count,
                max_in_bucket, flat_bucket_cumcount)

        # NOW WE WANT "BETWEEN" IMAGE PAIRS
        # 1) Get the vector indices of all (pure) neighbors of each atom
        # this gives \vb{g}(a, n) and f(a, n)
        # shapes 1 x A x Eta x 3 and 1 x A x Eta respectively
        # 
        # neighborhood count is A{n} (a), the number of atoms on the
        # neighborhood (all the neighbor buckets) of each atom,
        # A{n} (a) has shape 1 x A
        neighbor_bucket_vector_indices, neighbor_bucket_flat_indices =\
                self.get_neighbor_bucket_indices(main_bucket_vector_index)
        
        neighbor_flat_bucket_count = flat_bucket_count[neighbor_bucket_flat_indices] 
        neighbor_flat_bucket_cumcount = flat_bucket_cumcount[neighbor_bucket_flat_indices]
        neighborhood_count = neighbor_flat_bucket_count.sum(-1).squeeze() 

        # 2) Upper and lower part of the external pairlist
        # this is the correct "unpadded" upper part of the pairlist
        # it repeats each image idx a number of times equal to the number
        # of atoms on the neighborhood of each atom
        upper = torch.repeat_interleave(imidx_from_atidx.squeeze(), neighborhood_count)
        lower =\
            self.clist.get_lower_between_image_pairs(neighbor_flat_bucket_count,
                                   neighbor_flat_bucket_cumcount, max_in_bucket)
        assert lower.shape == upper.shape
        between_image_pairs = torch.stack((upper, lower), dim=0)
        print(between_image_pairs)
        print(within_image_pairs)
        exit()

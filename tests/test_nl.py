import torch
from torchani.modules import AEVComputerNL, CellListComputer
from torchani.geometry import tile_into_cube
import unittest

# comparison with self.coordinates
vector_bucket_index_compare =\
torch.tensor([[[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1],
         [0, 0, 1],
         [0, 0, 2],
         [0, 0, 2],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 1],
         [0, 1, 1],
         [0, 1, 2],
         [0, 1, 2],
         [0, 2, 0],
         [0, 2, 0],
         [0, 2, 1],
         [0, 2, 1],
         [0, 2, 2],
         [0, 2, 2],
         [1, 0, 0],
         [1, 0, 0],
         [1, 0, 1],
         [1, 0, 1],
         [1, 0, 2],
         [1, 0, 2],
         [1, 1, 0],
         [1, 1, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 1, 2],
         [1, 2, 0],
         [1, 2, 0],
         [1, 2, 1],
         [1, 2, 1],
         [1, 2, 2],
         [1, 2, 2],
         [2, 0, 0],
         [2, 0, 0],
         [2, 0, 1],
         [2, 0, 1],
         [2, 0, 2],
         [2, 0, 2],
         [2, 1, 0],
         [2, 1, 0],
         [2, 1, 1],
         [2, 1, 1],
         [2, 1, 2],
         [2, 1, 2],
         [2, 2, 0],
         [2, 2, 0],
         [2, 2, 1],
         [2, 2, 1],
         [2, 2, 2],
         [2, 2, 2]]], dtype=torch.long)


class TestCellList(unittest.TestCase):
    # Test that checks that the friendly constructor
    # reproduces the values from ANI1x with the correct parameters

    def setUp(self):
        cut = 5.2 - 0.01 # The length of the box is ~ 3 * this so that 
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 2.60001
        coordinates = torch.tensor([[cut/2, cut/2, cut/2],
                                     [cut/2 + 0.1, cut/2 + 0.1, cut/2 + 0.1]]).unsqueeze(0)
        species = torch.tensor([0, 0]).unsqueeze(0)
        species, coordinates = tile_into_cube((species, coordinates), box_length=cut)
        assert species.shape[1] == 54
        assert coordinates.shape[1] == 54
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        cell = torch.diag(torch.tensor([cut*3, cut*3, cut*3])).float()
        self.coordinates = coordinates
        self.species = species
        self.cell = cell

    def testInitDefault(self):
        clist = CellListComputer(5.2) # by default buckets_per_cutoff=1
        self.assertTrue(clist.buckets_per_cutoff==1)
        self.assertTrue(clist.cutoff == 5.2)
        self.assertTrue(clist.num_neighbors == 7)
        self.assertTrue((clist.bucket_length == 5.20001).all())

    def testInitTwo(self):
        clist = CellListComputer(5.2, buckets_per_cutoff=2)
        self.assertTrue(clist.buckets_per_cutoff==2)
        self.assertTrue(clist.cutoff == 5.2)
        self.assertTrue(clist.num_neighbors == 26)
        self.assertTrue((clist.bucket_length == 2.60001).all())

    def testSetupCell(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        # this creates a unit cell with 27 buckets
        # and a grid of 3 x 3 x 3 buckets (3 in each direction, Gx = Gy = Gz = 3)
        self.assertTrue(clist.total_buckets == 27)
        self.assertTrue((clist.shape_buckets_grid == torch.tensor([3,3,3], dtype=torch.long)).all())
        self.assertTrue(clist.vector_idx_to_flat.shape == torch.Size([3, 3, 3]))

    def testVectorIndexToFlat(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        # check some specific values of the tensor, it should be in row major
        # order so for instance the values in the z axis are 0 1 2 in the y
        # axis 0 3 6 and in the x axis 0 9 18
        self.assertTrue(clist.vector_idx_to_flat[0, 0, 0] == 0)
        self.assertTrue(clist.vector_idx_to_flat[1, 0, 0] == 9)
        self.assertTrue(clist.vector_idx_to_flat[0, 1, 0] == 3)
        self.assertTrue(clist.vector_idx_to_flat[0, 0, 1] == 1)
        self.assertTrue(clist.vector_idx_to_flat[-1, 0, 0] == 18)
        self.assertTrue(clist.vector_idx_to_flat[0, -1, 0] == 6)
        self.assertTrue(clist.vector_idx_to_flat[0, 0, -1] == 2)
        
    def testFractionalize(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        # test coordinate fractionalization
        frac = clist.fractionalize_coordinates(self.coordinates)
        self.assertTrue((frac < 1.0).all())

    def testVectorBucketIndex(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)

        frac = clist.fractionalize_coordinates(self.coordinates)
        main_vector_bucket_index = clist.fractional_to_vector_bucket_indices(frac)
        self.assertTrue(main_vector_bucket_index.shape == torch.Size([1, 54, 3]))
        self.assertTrue((main_vector_bucket_index == vector_bucket_index_compare).all())

    def testFlatBucketIndex(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        flat = clist.to_flat_index(vector_bucket_index_compare)
        # all flat bucket indices are present
        flat_compare = torch.repeat_interleave(torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testFlatBucketIndexAlternative(self):
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        atoms = vector_bucket_index_compare.shape[1]
        flat = clist.vector_idx_to_flat[vector_bucket_index_compare.reshape(-1, 3).unbind(1)].reshape(1, atoms)
        self.assertTrue(clist.total_buckets == 27)
        flat_compare = torch.repeat_interleave(torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testCounts(self):
        num_flat = 27
        clist = CellListComputer(5.2)
        clist.setup_cell_parameters(self.cell)
        flat = clist.to_flat_index(vector_bucket_index_compare)
        self.assertTrue(clist.total_buckets == num_flat)
        count_in_flat, cumcount_in_flat, max_ = clist.get_atoms_in_flat_bucket_counts(flat, num_flat)
        # these are all 2
        self.assertTrue(count_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((count_in_flat == 2).all())
        # these are all 0 2 4 6 ...
        self.assertTrue(cumcount_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((cumcount_in_flat == torch.arange(0, 54, 2)).all())
        # max counts in a bucket is 2
        self.assertTrue(max_ == 2)



    def testAEVComputerNLInit(self):
        AEVComputerNL.like_ani1x()
        
    #@unittest.skipIf(True, '')
    def testAEVComputerNL(self):
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        species, aevs = aevc((self.species, self.coordinates), cell=self.cell, pbc=pbc)


if __name__ == '__main__':
    unittest.main()

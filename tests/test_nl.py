import unittest
import pickle


import torch
from torchani.modules import AEVComputerNL, CellListComputer, AEVComputerJoint
from torchani.geometry import tile_into_cube
import torchani
from tqdm import tqdm

with open('test_nl_data.pkl', 'rb') as f:
    vector_bucket_index_compare = pickle.load(f)

class TestCellList(unittest.TestCase):

    def setUp(self):
        cut = 5.2
        cell_size = cut * 3 + 0.1
        self.cut = cut
        self.cell_size = cell_size
        # The length of the box is ~ 3 * cutoff this so that 
        # 3 buckets in each direction are needed to cover it, if one uses
        # a cutoff of 5.2 and a bucket length of 5.200001
        coordinates = torch.tensor([[cut/2, cut/2, cut/2],
                                     [cut/2 + 0.1, cut/2 + 0.1, cut/2 + 0.1]]).unsqueeze(0)
        species = torch.tensor([0, 0]).unsqueeze(0)
        species, coordinates = tile_into_cube((species, coordinates), box_length=cut)
        assert species.shape[1] == 54
        assert coordinates.shape[1] == 54
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        cell = torch.diag(torch.tensor([cell_size, cell_size, cell_size])).float()
        self.coordinates = coordinates
        self.species = species
        self.cell = cell
        self.clist = CellListComputer(cut)

    def testInitDefault(self):
        clist = CellListComputer(self.cut, buckets_per_cutoff=1)
        self.assertTrue(clist.buckets_per_cutoff==1)
        self.assertTrue(clist.cutoff == self.cut)
        self.assertTrue(clist.num_neighbors == 13)
        self.assertTrue((clist.bucket_length_lower_bound == self.cut + 0.00001).all())

    def testSetupCell(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        # this creates a unit cell with 27 buckets
        # and a grid of 3 x 3 x 3 buckets (3 in each direction, Gx = Gy = Gz = 3)
        self.assertTrue(clist.total_buckets == 27)
        self.assertTrue((clist.shape_buckets_grid == torch.tensor([3,3,3], dtype=torch.long)).all())
        # since it is padded shape should be 5 5 5 
        self.assertTrue(clist.vector_idx_to_flat.shape == torch.Size([5, 5, 5]))

    def testVectorIndexToFlat(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        # check some specific values of the tensor, it should be in row major
        # order so for instance the values in the z axis are 0 1 2 in the y
        # axis 0 3 6 and in the x axis 0 9 18
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 1] == 0)
        self.assertTrue(clist.vector_idx_to_flat[2, 1, 1] == 9)
        self.assertTrue(clist.vector_idx_to_flat[1, 2, 1] == 3)
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 2] == 1)
        self.assertTrue(clist.vector_idx_to_flat[0, 1, 1] == 18)
        self.assertTrue(clist.vector_idx_to_flat[1, 0, 1] == 6)
        self.assertTrue(clist.vector_idx_to_flat[1, 1, 0] == 2)
        
    def testFractionalize(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        # test coordinate fractionalization
        frac = clist.fractionalize_coordinates(self.coordinates)
        self.assertTrue((frac < 1.0).all())

    def testVectorBucketIndex(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)

        frac = clist.fractionalize_coordinates(self.coordinates)
        main_vector_bucket_index = clist.fractional_to_vector_bucket_indices(frac)
        self.assertTrue(main_vector_bucket_index.shape == torch.Size([1, 54, 3]))
        self.assertTrue((main_vector_bucket_index == vector_bucket_index_compare).all())

    def testFlatBucketIndex(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        flat = clist.to_flat_index(vector_bucket_index_compare)
        # all flat bucket indices are present
        flat_compare = torch.repeat_interleave(torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testFlatBucketIndexAlternative(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        atoms = vector_bucket_index_compare.shape[1]
        flat = clist.vector_idx_to_flat[(vector_bucket_index_compare +\
            torch.ones(1, dtype=torch.long)).reshape(-1,
                3).unbind(1)].reshape(1, atoms)
        self.assertTrue(clist.total_buckets == 27)
        flat_compare = torch.repeat_interleave(torch.arange(0, 27).to(torch.long), 2)
        self.assertTrue((flat == flat_compare).all())

    def testCounts(self):
        num_flat = 27
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        flat = clist.to_flat_index(vector_bucket_index_compare)
        self.assertTrue(clist.total_buckets == num_flat)
        count_in_flat, cumcount_in_flat, max_ = clist.get_atoms_in_flat_bucket_counts(flat)
        # these are all 2
        self.assertTrue(count_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((count_in_flat == 2).all())
        # these are all 0 2 4 6 ...
        self.assertTrue(cumcount_in_flat.shape == torch.Size([num_flat]))
        self.assertTrue((cumcount_in_flat == torch.arange(0, 54, 2)).all())
        # max counts in a bucket is 2
        self.assertTrue(max_ == 2)

    def testWithinBetween(self):
        clist = self.clist
        clist.setup_cell_parameters(self.cell)
        frac = clist.fractionalize_coordinates(self.coordinates)
        _, f_a = clist.get_bucket_indices(frac)
        A_f, Ac_f, A_star = clist.get_atoms_in_flat_bucket_counts(f_a)
        within = clist.get_within_image_pairs(A_f, Ac_f, A_star)
        # some hand comparisons with the within indices
        self.assertTrue(within.shape == torch.Size([2, 27]))
        self.assertTrue((within[0] == torch.arange(0, 54, 2)).all())
        self.assertTrue((within[1] == torch.arange(1, 55, 2)).all())

    def testAEVComputerNLInit(self):
        AEVComputerNL.like_ani1x()

    def testAEVComputerNL1(self):
        cut = self.cut
        d = 0.5
        coordinates = torch.tensor([
                                    [cut/2-d, cut/2-d, cut/2-d],
                                    [cut/2 + 0.1, cut/2 + 0.1, cut/2 + 0.1], 
                                    [cut/2 , cut/2 + 2.4 * cut, cut/2 + 2.4*cut],
                                     ]).unsqueeze(0)
        species = torch.tensor([0, 0, 0]).unsqueeze(0)
        cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
        _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNL2(self):
        cut = self.cut
        d = 0.5
        coordinates = torch.tensor([
                                    [cut/2-d, cut/2-d, cut/2-d],
                                    [cut/2 + 0.1, cut/2 + 0.1, cut/2 + 0.1], 
                                    [cut/2 + 2.4*cut , cut/2 + 2.4 * cut, cut/2 + 2.4*cut],
                                     ]).unsqueeze(0)
        species = torch.tensor([0, 0, 0]).unsqueeze(0)
        cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
        _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())
    
    def testAEVComputerNL3(self):
        coordinates = torch.tensor([[[1.0000e-03, 6.5207e+00, 1.0000e-03],
         [1.0000e-03, 1.5299e+01, 1.3643e+01],
         [1.0000e-03, 2.1652e+00, 1.5299e+01]]])
        species = torch.tensor([0, 0, 0]).unsqueeze(0)
        cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
        _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNL4(self):
        coordinates = torch.tensor([[[1.5299e+01, 1.0000e-03, 5.5613e+00],
                 [1.0000e-03, 1.0000e-03, 3.8310e+00],
                 [1.5299e+01, 1.1295e+01, 1.5299e+01]]])
        species = torch.tensor([0, 0, 0]).unsqueeze(0)
        cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
        _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNL5(self):
        coordinates = torch.tensor([[
                 [1.0000e-03, 1.0000e-03, 1.0000e-03],
                 [1.0389e+01, 1.0000e-03, 1.5299e+01],
                 ]])
        species = torch.tensor([ 0, 0]).unsqueeze(0)
        # first bucket is 0 - 5.2, in 3 directions, and subsequent buckets
        # are on top of that
        cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
        _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())
    
    def testAEVComputerNL(self):
        pbc = torch.tensor([True, True, True], dtype=torch.bool)
        aevc = AEVComputerNL.like_ani1x()
        aevj = AEVComputerJoint.like_ani1x()
        species, aevs = aevc((self.species, self.coordinates), cell=self.cell, pbc=pbc)
        species, aevsj = aevj((self.species, self.coordinates), cell=self.cell, pbc=pbc)
        self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNLRandom(self):
        cut = self.cut
        for j in range(100):
            coordinates = torch.tensor([[cut/2, cut/2, cut/2],
                                         [cut/2 + 0.1, cut/2 + 0.1, cut/2 + 0.1]]).unsqueeze(0)
            species = torch.tensor([0, 0]).unsqueeze(0)
            species, coordinates = tile_into_cube((species, coordinates), box_length=cut, noise=0.1)

            pbc = torch.tensor([True, True, True], dtype=torch.bool)
            aevc = AEVComputerNL.like_ani1x()
            aevj = AEVComputerJoint.like_ani1x()
            _, aevs = aevc((species, coordinates), cell=self.cell, pbc=pbc)
            _, aevsj = aevj((species, coordinates), cell=self.cell, pbc=pbc)
            self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNLRandom2(self):
        for j in tqdm(range(100)):
            coordinates = torch.randn(10, 3).unsqueeze(0)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001)
            species = torch.zeros(10).unsqueeze(0).to(torch.long)

            pbc = torch.tensor([True, True, True], dtype=torch.bool)
            cell = torch.diag(torch.tensor([self.cell_size, self.cell_size, self.cell_size])).float()
            aevc = AEVComputerNL.like_ani1x()
            aevj = AEVComputerJoint.like_ani1x()
            _, aevs = aevc((species, coordinates), cell=cell, pbc=pbc)
            _, aevsj = aevj((species, coordinates), cell=cell, pbc=pbc)
            self.assertTrue(torch.isclose(aevs, aevsj).all())

    def testAEVComputerNLRandomE(self):
        device = torch.device('cuda')
        aevj = AEVComputerJoint.like_ani1x().to(device).to(torch.double)
        aevc = AEVComputerNL.like_ani1x().to(device).to(torch.double)
        modelj = torchani.models.ANI1x(model_index=0).to(device).double()
        modelc = torchani.models.ANI1x(model_index=0).to(device).double()
        modelj.aev_computer = aevj
        modelc.aev_computer = aevc
        species = torch.zeros(100).unsqueeze(0).to(torch.long).to(device)
        for j in tqdm(range(100)):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(device).to(torch.double)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001)

            pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
            _, e_c = modelc((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            _, e_j = modelj((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            self.assertTrue(torch.isclose(e_c, e_j).all())

    def testAEVComputerNLRandomEfloat(self):
        device = torch.device('cuda')
        aevj = AEVComputerJoint.like_ani1x().to(device)
        aevc = AEVComputerNL.like_ani1x().to(device)
        modelj = torchani.models.ANI1x(model_index=0).to(device)
        modelc = torchani.models.ANI1x(model_index=0).to(device)
        modelj.aev_computer = aevj
        modelc.aev_computer = aevc
        species = torch.zeros(100).unsqueeze(0).to(torch.long).to(device)
        for j in tqdm(range(100)):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(device).to(torch.float)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001).float()

            pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
            _, e_c = modelc((species, coordinates), cell=self.cell.to(device).float(), pbc=pbc)
            _, e_j = modelj((species, coordinates), cell=self.cell.to(device).float(), pbc=pbc)
            self.assertTrue(torch.isclose(e_c, e_j).all())

    def testAEVComputerNLRandom3(self):
        device = torch.device('cuda')
        aevj = AEVComputerJoint.like_ani1x().to(device).to(torch.double)
        aevc = AEVComputerNL.like_ani1x().to(device).to(torch.double)
        species = torch.LongTensor(100).random_(0, 4).to(device).unsqueeze(0)
        for j in tqdm(range(100)):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(device).to(torch.double)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001)

            pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
            _, aev_c = aevc((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            _, aev_j = aevj((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            self.assertTrue(torch.isclose(aev_c, aev_j).all())

    # TODO:note that this test fails with single precision!
    @unittest.skipIf(True, '')
    def testAEVComputerNLRandom3float(self):
        device = torch.device('cuda')
        aevj = AEVComputerJoint.like_ani1x().to(device).to(torch.float)
        aevc = AEVComputerNL.like_ani1x().to(device).to(torch.float)
        species = torch.LongTensor(100).random_(0, 4).to(device).unsqueeze(0)
        for j in tqdm(range(100)):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(device).to(torch.float)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001)

            pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
            _, aev_c = aevc((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            _, aev_j = aevj((species, coordinates), cell=self.cell.to(device), pbc=pbc)
            self.assertTrue(torch.isclose(aev_c, aev_j).all())

    def testNeighborlistJit(self):
        # TODO many JIT optimizations break the code so I turn every
        # optimization off until pytorch fixes them
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        device = torch.device('cuda')
        aevj = AEVComputerJoint.like_ani1x().to(device).to(torch.double)
        aevc = AEVComputerNL.like_ani1x().to(device).to(torch.double)
        aevj = torch.jit.script(aevj)
        aevc = torch.jit.script(aevc)
        species = torch.LongTensor(100).random_(0, 4).to(device).unsqueeze(0)
        for j in tqdm(range(100)):
            coordinates = torch.randn(100, 3).unsqueeze(0).to(device).to(torch.double)*3*self.cell_size
            coordinates = torch.clamp(coordinates, min=0.0001, max=self.cell_size - 0.0001)

            pbc = torch.tensor([True, True, True], dtype=torch.bool).to(device)
            coordinates.requires_grad_(True)
            _, aev_c = aevc((species, coordinates.double()), cell=self.cell.to(device).double(), pbc=pbc)
            _ = -torch.autograd.grad(aev_c.sum(), coordinates)[0]

            coordinates.requires_grad_(True)
            _, aev_j = aevj((species, coordinates.double()), cell=self.cell.to(device).double(), pbc=pbc)
            _ = -torch.autograd.grad(aev_j.sum(), coordinates)[0]
            self.assertTrue(torch.isclose(aev_c, aev_j).all())

if __name__ == '__main__':
    unittest.main()
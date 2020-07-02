import torch
import torchani
import unittest
#  current unsupported aten operators in opset 12 (ONNX 1.7.0):
# -torch.Tensor.index_add
# -torch.Tensor.unique_consecutive

# -torch.triu_indices done
# -torch.BoolTensor.any done
# -torch.repeat_interleave done
# -For opset 11, which is ONNX 1.6.0, also:
# -torch.nn.functional.celu done


class ForcesModel(torch.nn.Module):
    # TensorRT does NOT have a builtin autograd
    # TODO: check if this has any chance of working or is just fantasy
    # if ONNX doesn't have a builtin autograd something like this may be
    # necessary in order to calculate forces, this is a model that internally
    # performs autograd inside forward, although I suspect this will not work
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_):
        species, coordinates = input_
        energies = self.model((species, coordinates)).energies
        energies = energies.sum()
        forces = -torch.autograd.grad(energies, coordinates)[0]
        return forces


class TestONNX(unittest.TestCase):
    # tracing tests are currently performed falling back on aten operators for
    # opset 11 which is the version supported by NVIDIA TensorRT to determine
    # which operators are unsupported and which ones are supported WARNING:
    # tracing a model for onnx means that the model will be fixed for one
    # molecule type, which is not ideal but this is done as a benchmarking
    # exercise for the moment
    def setUp(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.coordinates = torch.tensor(
            [[[0.03192167, 0.00638559, 0.01301679],
              [-0.83140486, 0.39370209, -0.26395324],
              [-0.66518241, -0.84461308, 0.20759389],
              [0.45554739, 0.54289633, 0.81170881],
              [0.66091919, -0.16799635, -0.91037834]],
             [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
              [0., 0., 0.]]],
            requires_grad=True,
            device=self.device)
        self.species = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0]],
                                    dtype=torch.long,
                                    device=self.device)
        self.model = torchani.models.ANI1x(periodic_table_index=False,
                                           model_index=0,
                                           onnx_opset11=True).to(self.device)
        self.prefix_for_onnx_files = ''

    @unittest.skipIf(True, 'always')
    def testForcesTrace(self):
        forces_model = ForcesModel(self.model).to(self.device)
        example_outputs = forces_model((self.species, self.coordiantes))

        torch.onnx.export(forces_model, ((self.species, self.coordinates), ),
                          f'{self.prefix_for_onnx_files}forces_model.onnx',
                          verbose=True,
                          opset_version=11,
                          example_outputs=example_outputs,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_ATEN_FALLBACK)

    @unittest.skipIf(True, 'skip')
    def testANIModelTrace(self):
        # checks if ANIModel is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        ani1x = self.model
        ani_model = ani1x.neural_networks
        species, aevs = ani1x.aev_computer((self.species, self.coordinates))
        example_outputs = ani_model((species, aevs))
        torch.onnx.export(ani_model, ((species, aevs), ),
                          f'{self.prefix_for_onnx_files}ani_model.onnx',
                          example_outputs=example_outputs,
                          verbose=True,
                          opset_version=11)

    # @unittest.skipIf(True, 'skip')
    def testEnergyShifterTrace(self):
        # checks if EnergyShifter is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        ani1x = self.model
        energy_shifter = ani1x.energy_shifter
        species, energies = torchani.nn.Sequential(ani1x.aev_computer,
                                                   ani1x.neural_networks)(
                                                       (self.species,
                                                        self.coordinates))
        example_outputs = energy_shifter((species, energies))
        torch.onnx.export(energy_shifter, ((species, energies), ),
                          f'{self.prefix_for_onnx_files}energy_shifter.onnx',
                          verbose=True,
                          example_outputs=example_outputs,
                          opset_version=11)

    @unittest.skipIf(True, 'skip')
    def testAEVComputerTrace(self):
        # checks if AEVComputer() is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        ani1x = self.model
        aev_computer = ani1x.aev_computer
        example_outputs = aev_computer((self.species, self.coordiantes))
        torch.onnx.export(aev_computer, ((self.species, self.coordinates), ),
                          f'{self.prefix_for_onnx_files}aev_computer.onnx',
                          verbose=True,
                          example_outputs=example_outputs,
                          opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_ATEN_FALLBACK)


class TestScriptModuleONNX(TestONNX):
    # Tests ScriptModule exports instead of plain traces
    def setUp(self):
        super().setUp()
        self.model = torch.jit.script(self.model)
        self.prefix_for_onnx_files = 'jit_'


if __name__ == '__main__':
    unittest.main()

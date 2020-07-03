import torch
from torch import Tensor
import torchani
import unittest
import onnx
#  current unsupported aten operators in opset 12 (ONNX 1.7.0):
# -torch.Tensor.index_add
# -torch.Tensor.unique_consecutive
# -torch.BoolTensor.any needs to be added for arbitrary dimensions

# -torch.triu_indices done
# -torch.repeat_interleave done
# -For opset 11, which is ONNX 1.6.0, also:
# -torch.nn.functional.celu done


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, species: Tensor, input_: Tensor):
        species = self.model((species, input_))[0]
        out = self.model((species, input_))[1]
        return species, out


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


class TestTraceONNX(unittest.TestCase):
    # tracing tests are currently performed for AEVComputer by falling back on aten operators for
    # opset 11 which is the version supported by NVIDIA TensorRT to determine
    # which operators are unsupported and which ones are supported

    # WARNING:
    # tracing a model for onnx means that the model will be fixed for one
    # molecule type, which is not ideal but this is done as a benchmarking
    # exercise for the moment

    # checks if ANIModel, EnergyShifter, etc are onnx-traceable
    # currently only checks gross RuntimeErrors when tracing and checks if
    # the resulting model intermediate representation is well formed

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
    def testForces(self):
        forces_model = ForcesModel(self.model).to(self.device)
        example_outputs = forces_model((self.species, self.coordiantes))

        torch.onnx.export(forces_model, ((self.species, self.coordinates), ),
                          f'{self.prefix_for_onnx_files}forces_model.onnx',
                          verbose=True,
                          opset_version=11,
                          example_outputs=example_outputs,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_ATEN_FALLBACK)

    @unittest.skipIf(True, 'always')
    def testANIModel(self):
        ani_model = ModelWrapper(self.model.neural_networks)
        self._testANIModel(ani_model)

    @unittest.skipIf(True, 'always')
    def testEnergyShifter(self):
        energy_shifter = ModelWrapper(self.model.energy_shifter)
        self._testEnergyShifter(energy_shifter)

    def _testANIModel(self, ani_model):
        species, aevs = self.model.aev_computer(
            (self.species, self.coordinates))

        example_outputs = ani_model(species, aevs)
        # no use for names and Dynamic axes when tracing
        torch.onnx.export(ani_model, (
            species,
            aevs,
        ),
                          f'{self.prefix_for_onnx_files}ani_model.onnx',
                          input_names=['species', 'aevs'],
                          output_names=['species_out', 'unshifted_energies'],
                          dynamic_axes={
                              'species': {
                                  0: 'conformations',
                                  1: 'atoms'
                              },
                              'aevs': {
                                  0: 'conformations',
                                  1: 'atoms'
                              },
                              'species_out': {
                                  0: 'conformations',
                                  1: 'atoms'
                              },
                              'unshifted_energies': {
                                  0: 'conformations'
                              }
                          },
                          example_outputs=example_outputs,
                          verbose=True, 
                          opset_version=11)
        model_onnx = onnx.load(f'{self.prefix_for_onnx_files}ani_model.onnx')
        onnx.checker.check_model(model_onnx)

    def _testEnergyShifter(self, energy_shifter):
        # checks if EnergyShifter is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing

        species, energies = torchani.nn.Sequential(self.model.aev_computer,
                                                   self.model.neural_networks)(
                                                       (self.species,
                                                        self.coordinates))
        example_outputs = energy_shifter(species, energies)
        # no use for names and Dynamic axes when tracing
        torch.onnx.export(energy_shifter, (species, energies),
                          f'{self.prefix_for_onnx_files}energy_shifter.onnx',
                          input_names=['species', 'unshifted_energies'],
                          output_names=['species_out', 'shifted_energies'],
                          dynamic_axes={
                              'species': {
                                  0: 'conformations',
                                  1: 'atoms'
                              },
                              'unshifted_energies': {
                                  0: 'conformations'
                              },
                              'species_out': {
                                  0: 'conformations',
                                  1: 'atoms'
                              },
                              'shifted_energies': {
                                  0: 'conformations'
                              }
                          },
                          example_outputs=example_outputs,
                          opset_version=11)
        model_onnx = onnx.load(f'{self.prefix_for_onnx_files}energy_shifter.onnx')
        onnx.checker.check_model(model_onnx)

    @unittest.skipIf(True, 'skip')
    def testAEVComputer(self):
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


class TestScriptModuleONNX(TestTraceONNX):
    # checks if ANIModel, EnergyShifter, etc are onnx-exportable as
    # ScriptModules currently only checks gross RuntimeErrors when tracing and
    # checks if the resulting model intermediate representation is well formed
    def setUp(self):
        super().setUp()
        self.prefix_for_onnx_files = 'jit_'

    @unittest.skipIf(True, 'always')
    def testEnergyShifter(self):
        # checks if EnergyShifter is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        energy_shifter = ModelWrapper(self.model.energy_shifter)
        energy_shifter = torch.jit.script(energy_shifter)
        self._testEnergyShifter(energy_shifter)


    def testANIModel(self):
        # checks if ANIModel is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        # TODO: The exported graph for ANIModel is very wrong 
        # right now, that is possibly because of the 
        # 'for i, (_, m) in enumerate(self.items()) loop'
        # which it seems like it is not registered at all by 
        # onnx (but it is if the model is traced)
        ani_model = ModelWrapper(self.model.neural_networks)
        ani_model = torch.jit.script(ani_model)
        self._testANIModel(ani_model)


if __name__ == '__main__':
    unittest.main()

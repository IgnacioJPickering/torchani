import torch
from torch import Tensor
import torchani
import unittest
import numpy as np
import os

# used for explicit graph checking
import onnx

# Microsoft backend used to validate runtime forward passes of exported models
import onnxruntime as ort

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
        return out


class ForcesModel(torch.nn.Module):
    # TensorRT does NOT have a builtin autograd
    # There doesn't seem to be a way to make this work, either with 
    # torch.autograd.grad or Tensor.backward(). Both cases fail since
    # JIT fails for models that have autograd internally, and 
    # both tracing and scripting need JIT
    #
    # only workaround seems to be writing down derivatives explicitly
    #
    # TODO: check if this has any chance of working or is just fantasy
    # if ONNX doesn't have a builtin autograd something like this may be
    # necessary in order to calculate forces, this is a model that internally
    # performs autograd inside forward, although I suspect this will not work
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, species, coordinates):
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

    # checks if ANIModel, EnergyShifter, Ensemble, etc are onnx-traceable
    # currently this test only checks gross RuntimeErrors when tracing and
    # checks if the resulting model intermediate representation is well formed

    # It is necessary to wrap the model inside a Module that
    # does not accept or output tuples, since it seems like this can screw
    # up ONNX exporting in various ways. Support for non tensor inputs /
    # outputs is not very good currently

    def setUp(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.coordinates = torch.tensor(
            [[[0.03192167, 0.00638559, 0.01301679],
              [-0.83140486, 0.39370209, -0.26395324],
              [-0.66518241, -0.84461308, 0.20759389],
              [0.45554739, 0.54289633, 0.81170881],
              [0.66091919, -0.16799635, -0.91037834]],
             [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
              [0., 0., 0.]]],
            requires_grad=True,
            device=self.device,
            dtype=torch.float)
        self.species = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0]],
                                    dtype=torch.long,
                                    device=self.device)
        self.np_coordinates = self.coordinates.clone().detach().cpu().numpy(
        ).astype(np.float32)
        self.np_species = self.species.clone().detach().cpu().numpy().astype(
            np.long)
        self.ensemble_model = torchani.models.ANI1x(
            periodic_table_index=False,
            onnx_opset11=True).to(self.device).to(torch.float)
        self.model = self.ensemble_model[0]
        self.prefix_for_onnx_files = ''

    @unittest.skipIf(True, 'skip')
    def testForces(self):
        forces_model = ForcesModel(self.model).to(self.device)
        example_outputs = forces_model(self.species, self.coordinates)
        torch.onnx.export(forces_model, (self.species, self.coordinates ),
                          f'{self.prefix_for_onnx_files}forces_model.onnx',
                          verbose=True,
                          opset_version=11,
                          example_outputs=example_outputs,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_ATEN_FALLBACK)

    @unittest.skipIf(True, 'skip')
    def testAEVComputer(self):
        # checks if AEVComputer() is onnx-traceable
        # currently only checks gross RuntimeErrors when tracing
        ani1x = self.model
        aev_computer = ani1x.aev_computer
        example_outputs = aev_computer((self.species, self.coordinates))
        torch.onnx.export(aev_computer, ((self.species, self.coordinates), ),
                          f'{self.prefix_for_onnx_files}aev_computer.onnx',
                          verbose=True,
                          example_outputs=example_outputs,
                          opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.
                          ONNX_ATEN_FALLBACK)

    def testEnergyShifter(self):
        energy_shifter = ModelWrapper(self.model.energy_shifter)
        self._testEnergyShifter(energy_shifter, 'energy_shifter')

    def testSequential(self):
        sequential = ModelWrapper(
            torchani.nn.Sequential(self.model.neural_networks,
                                   self.model.energy_shifter))
        self._testNNModuleCommon(sequential,
                                 'sequential',
                                 output_shifted_energies=True)

    def testANIModel(self):
        nn_module = ModelWrapper(self.model.neural_networks)
        self._testNNModuleCommon(nn_module, 'ani_model')

    def testEnsemble(self):
        nn_module = ModelWrapper(self.ensemble_model.neural_networks)
        self._testNNModuleCommon(nn_module, 'ensemble')

    def _testEnergyShifter(self, module, name):
        # This function is used to test both torchani.EnergyShifter
        # module must already be wrapped
        input_names = ['species', 'unshifted_energies']
        output_names = ['energies']
        # C : conformations (batch dimension), A : atoms
        dynamic_axes = {
            input_names[0]: {
                0: 'C',
                1: 'A'
            },
            input_names[1]: {
                0: 'C'
            },
            output_names[0]: {
                0: 'C'
            }
        }
        energies, np_energies = self._getUnshiftedEnergies()
        onnx_file_name = f'{self.prefix_for_onnx_files}{name}.onnx'
        example_outputs = self._testExportModel(module, input_names,
                                                output_names, dynamic_axes,
                                                onnx_file_name,
                                                (self.species, energies))
        self._testONNXGraph(onnx_file_name)
        self._testONNXRuntime(self.np_species, np_energies, onnx_file_name,
                              example_outputs)
        # remove created onnx file to avoid polluting the tests tree
        os.unlink(onnx_file_name)

    def _testNNModuleCommon(self,
                            nn_module,
                            name,
                            output_shifted_energies=False):
        # This function is used to test both torchani.Ensemble and
        # torchani.ANIModule since they expose the same API
        # nn_module must already be wrapped
        input_names = ['species', 'aevs']
        if output_shifted_energies:
            output_names = ['energies']
        else:
            output_names = ['unshifted_energies']
        # C : conformations (batch dimension), A : atoms
        dynamic_axes = {
            input_names[0]: {
                0: 'C',
                1: 'A'
            },
            input_names[1]: {
                0: 'C',
                1: 'A'
            },
            output_names[0]: {
                0: 'C'
            }
        }
        aevs, np_aevs = self._getAEVs()
        onnx_file_name = f'{self.prefix_for_onnx_files}{name}.onnx'
        example_outputs = self._testExportModel(nn_module, input_names,
                                                output_names, dynamic_axes,
                                                onnx_file_name,
                                                (self.species, aevs))
        self._testONNXGraph(onnx_file_name)
        self._testONNXRuntime(self.np_species, np_aevs, onnx_file_name,
                              example_outputs)
        os.unlink(onnx_file_name)

    def _getAEVs(self):
        species, aevs = self.model.aev_computer(
            (self.species, self.coordinates))
        np_aevs = aevs.clone().detach().cpu().numpy()
        return aevs, np_aevs

    def _getUnshiftedEnergies(self):
        species, energies = torchani.nn.Sequential(self.model.aev_computer,
                                                   self.model.neural_networks)(
                                                       (self.species,
                                                        self.coordinates))
        np_energies = energies.clone().detach().cpu().numpy()
        return energies, np_energies

    def _testExportModel(self, module, input_names, output_names, dynamic_axes,
                         onnx_file_name, inputs_):
        species, other_input = inputs_
        example_outputs = module(species, other_input)
        # names are added to inputs and parameters only for easier
        # understanding of the resulting onnx graph
        for k, p in module.named_parameters():
            input_names.append(k)
        torch.onnx.export(module, (species, other_input),
                          onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          #verbose=True,
                          example_outputs=example_outputs,
                          opset_version=11)
        return example_outputs

    @staticmethod
    def _testONNXGraph(onnx_file_name):
        # checks the graph using the builtin ONNX Checker
        model_onnx = onnx.load(onnx_file_name)
        onnx.checker.check_model(model_onnx)

    def _testONNXRuntime(self, input0, input1, onnx_file_name,
                         example_outputs):
        # check that model can be run by using Microsoft ONNX Runtime backend
        ort_session = ort.InferenceSession(onnx_file_name)
        input_name0 = ort_session.get_inputs()[0].name
        input_name1 = ort_session.get_inputs()[1].name
        output_name = ort_session.get_outputs()[0].name
        output = ort_session.run([output_name], {
            input_name0: input0,
            input_name1: input1
        })
        example_outputs = example_outputs.detach().cpu().numpy()
        # check that outputs from the model is similar to output
        # from torchani
        self.assertTrue(np.isclose(example_outputs, output[0]).all())


class TestScriptModuleONNX(TestTraceONNX):
    # checks if ANIModel, EnergyShifter, etc are onnx-exportable as
    # ScriptModules currently only checks gross RuntimeErrors when tracing and
    # checks if the resulting model intermediate representation is well formed
    def setUp(self):
        super().setUp()
        self.prefix_for_onnx_files = 'jit_'

    def testSequential(self):
        sequential = ModelWrapper(
            torchani.nn.Sequential(self.model.neural_networks,
                                   self.model.energy_shifter))
        sequential = torch.jit.script(sequential)
        self._testNNModuleCommon(sequential,
                                 'sequential',
                                 output_shifted_energies=True)

    def testANIModel(self):
        nn_module = ModelWrapper(self.model.neural_networks)
        nn_module = torch.jit.script(nn_module)
        self._testNNModuleCommon(nn_module, 'ani_model')

    def testEnsemble(self):
        nn_module = ModelWrapper(self.ensemble_model.neural_networks)
        nn_module = torch.jit.script(nn_module)
        self._testNNModuleCommon(nn_module, 'ensemble')

    def testEnergyShifter(self):
        energy_shifter = ModelWrapper(self.model.energy_shifter)
        energy_shifter = torch.jit.script(energy_shifter)
        self._testEnergyShifter(energy_shifter, 'energy_shifter')


if __name__ == '__main__':
    unittest.main()

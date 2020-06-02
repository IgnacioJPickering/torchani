# -*- coding: utf-8 -*-
"""The ANI model zoo that stores public ANI models.

Currently the model zoo has two models: ANI-1x and ANI-1ccx. The classes
of these two models are :class:`ANI1x` and :class:`ANI1ccx`,
these are subclasses of :class:`torch.nn.Module`.
To use the models just instantiate them and either
directly calculate energies or get an ASE calculator. For example:

.. code-block:: python

    ani1x = torchani.models.ANI1x()
    # compute energy using ANI-1x model ensemble
    _, energies = ani1x((species, coordinates))
    ani1x.ase()  # get ASE Calculator using this ensemble
    # convert atom species from string to long tensor
    ani1x.species_to_tensor('CHHHH')

    model0 = ani1x[0]  # get the first model in the ensemble
    # compute energy using the first model in the ANI-1x model ensemble
    _, energies = model0((species, coordinates))
    model0.ase()  # get ASE Calculator using this model
    # convert atom species from string to long tensor
    model0.species_to_tensor('CHHHH')

Note that the class BuiltinModels can be accessed but it is deprecated and
shouldn't be used anymore.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
from pkg_resources import resource_filename
from . import neurochem
from .nn import SpeciesConverter, SpeciesEnergies
from .aev import AEVComputer


class BuiltinModel(torch.nn.Module):
    r"""Private template for the builtin ANI models """
    def __init__(self, species_converter, aev_computer, neural_networks, energy_shifter, species_to_tensor, periodic_table_index):
        if periodic_table_index:
            self.species_converter = species_converter
        else:
            self.species_converter = None
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter
        self._species_to_tensor = species_to_tensor
        self.periodic_table_index = periodic_table_index

    def _from_neurochem_resources(cls, info_file_path, index=0, periodic_table_index=False):
        # Convenience function for building only one model from the ensemble

        def get_resource(file_path, package_name):
            return resource_filename(package_name, 'resources/' + file_path)

        package_name = '.'.join(__name__.split('.')[:-1])
        info_file = get_resource(info_file_path, package_name)

        with open(info_file) as f:
            # const_file: Path to the file with the builtin constants.
            # sae_file: Path to the file with the Self Atomic Energies.
            # ensemble_prefix: Prefix of the neurochem resource directories.
            lines = [x.strip() for x in f.readlines()][:4]
            const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
            const_file = get_resource(const_file_path, package_name)
            sae_file = get_resource(sae_file_path, package_name)
            sae_file = get_resource(sae_file_path, package_name)
            ensemble_prefix = resource_filename(package_name,
                                                ensemble_prefix_path)
            ensemble_size = int(ensemble_size)
            consts = neurochem.Constants(const_file)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)
        energy_shifter, _ = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, periodic_table_index)

    @torch.jit.export
    def _recast_long_buffers(self):
        self.species_converter.conv_tensor = self.species_converter.conv_tensor.to(dtype=torch.long)
        self.aev_computer.triu_index = self.aev_computer.triu_index.to(dtype=torch.long)

    def forward(self, species_coordinates: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """Calculates predicted properties for minibatch of configurations

        Args:
            species_coordinates: minibatch of configurations
            cell: the cell used in PBC computation, set to None if PBC is not enabled
            pbc: the bool tensor indicating which direction PBC is enabled, set to None if PBC is not enabled

        Returns:
            species_energies: energies for the given configurations

        .. note:: The coordinates, and cell are in Angstrom, and the energies
            will be in Hartree.
        """
        if self.periodic_table_index:
            species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates, cell=cell, pbc=pbc)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)

    def species_to_tensor(self, *args, **kwargs):
        """Convert species from strings to tensor.

        See also :method:`torchani.neurochem.Constant.species_to_tensor`

        Arguments:
            species (:class:`str`): A string of chemical symbols

        Returns:
            tensor (:class:`torch.Tensor`): A 1D tensor of integers
        """
        # The only difference between this and the "raw" private version
        # _species_to_tensor is that this sends the final tensor to the model
        # device
        return self._species_to_tensor(*args, **kwargs) \
            .to(self.aev_computer.ShfR.device)

    def ase(self, **kwargs):
        """Get an ASE Calculator using this ANI model

        Arguments:
            kwargs: ase.Calculator kwargs

        Returns:
            calculator (:class:`int`): A calculator to be used with ASE
        """
        from . import ase
        return ase.Calculator(self.species, self, **kwargs)


class BuiltinEnsemble(BuiltinModel):
    """Private template for the builtin ANI ensemble models.

    ANI ensemble models form the ANI models zoo are instances of this class.
    This class is a torch module that sequentially calculates
    AEVs, then energies from a torchani.Ensemble and then uses EnergyShifter
    to shift those energies. It is essentially a sequential

    'AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=False), or a sequential

    'SpeciesConverter -> AEVComputer -> Ensemble -> EnergyShifter'

    (periodic_table_index=True).

    .. note::
        This class is for internal use only, avoid relying on anything from it
        except the public methods, always use ANI1x, ANI1ccx, etc to instance
        the models.
        Also, don't confuse this class with torchani.Ensemble, which is only a
        container for many ANIModel instances and shouldn't be used directly
        for calculations.

    Attributes:
        species_converter (:class:`torchani.nn.SpeciesConverter`): Converts periodic table index to
            internal indices. Only present if periodic_table_index is `True`.
        aev_computer (:class:`torchani.AEVComputer`): AEV computer with
            builtin constants
        energy_shifter (:class:`torchani.EnergyShifter`): Energy shifter with
            builtin Self Atomic Energies.
        periodic_table_index (bool): Whether to use element number in periodic table
            to index species. If set to `False`, then indices must be `0, 1, 2, ..., N - 1`
            where `N` is the number of parametrized species.
    """

    def __init__(self, species_converter, aev_computer, neural_networks,
                 energy_shifter, species_to_tensor, periodic_table_index):
        super(BuiltinEnsemble, self).__init__(species_converter,
                                              aev_computer,
                                              neural_networks,
                                              energy_shifter,
                                              species_to_tensor,
                                              periodic_table_index)

    @classmethod
    def _from_neurochem_resources(cls, info_file_path, periodic_table_index=False):

        def get_resource(file_path):
            return resource_filename(package_name, 'resources/' + file_path)

        package_name = '.'.join(__name__.split('.')[:-1])
        info_file = get_resource(info_file_path)

        with open(info_file) as f:
            # const_file: Path to the file with the builtin constants.
            # sae_file: Path to the file with the Self Atomic Energies.
            # ensemble_prefix: Prefix of the neurochem resource directories.
            lines = [x.strip() for x in f.readlines()][:4]
            const_file_path, sae_file_path, ensemble_prefix_path, ensemble_size = lines
            const_file = get_resource(const_file_path)
            sae_file = get_resource(sae_file_path)
            sae_file = get_resource(sae_file_path)
            ensemble_prefix = resource_filename(package_name,
                                                ensemble_prefix_path)
            ensemble_size = int(ensemble_size)
            consts = neurochem.Constants(const_file)

        species_converter = SpeciesConverter(consts.species)
        aev_computer = AEVComputer(**consts)
        neural_networks = neurochem.load_model_ensemble(consts.species,
                                                        ensemble_prefix, ensemble_size)
        energy_shifter, _ = neurochem.load_sae(sae_file, return_dict=True)
        species_to_tensor = consts.species_to_tensor

        return cls(species_converter, aev_computer, neural_networks,
                   energy_shifter, species_to_tensor, periodic_table_index)

    def __getitem__(self, index):
        """Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model

        Get a single 'AEVComputer -> ANIModel -> EnergyShifter' sequential model
        or
        Indexing allows access to a single model inside the ensemble
        that can be used directly for calculations. The model consists
        of a sequence AEVComputer -> ANIModel -> EnergyShifter
        and can return an ase calculator and convert species to tensor.

        Args:
            index (:class:`int`): Index of the model

        Returns:
            ret: (:class:`torchani.models.BuiltinModel`) Model ready for
                calculations
        """
        ret = BuiltinModel(self.species_converter, self.aev_computer,
                           self.neural_networks[index], self.energy_shifter,
                           self.species_to_tensor_raw, self.periodic_table_index)
        return ret

    def __len__(self):
        """Get the number of networks in the ensemble

        Returns:
            length (:class:`int`): Number of networks in the ensemble
        """
        return len(self.neural_networks)


def ANI1x(periodic_table_index=False):
    """The ANI-1x model as in `ani-1x_8x on GitHub`_ and `Active Learning Paper`_.

    The ANI-1x model is an ensemble of 8 networks that was trained using
    active learning on the ANI-1x dataset, the target level of theory is
    wB97X/6-31G(d). It predicts energies on HCNO elements exclusively, it
    shouldn't be used with other atom types.

    .. _ani-1x_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1x_8x

    .. _Active Learning Paper:
        https://aip.scitation.org/doi/abs/10.1063/1.5023802
    """
    return BuiltinEnsemble._from_neurochem_resources('ani-1x_8x.info', periodic_table_index)


def ANI1ccx(periodic_table_index=False):
    """The ANI-1ccx model as in `ani-1ccx_8x on GitHub`_ and `Transfer Learning Paper`_.

    The ANI-1ccx model is an ensemble of 8 networks that was trained
    on the ANI-1ccx dataset, using transfer learning. The target accuracy
    is CCSD(T)*/CBS (CCSD(T) using the DPLNO-CCSD(T) method). It predicts
    energies on HCNO elements exclusively, it shouldn't be used with other
    atom types.

    .. _ani-1ccx_8x on GitHub:
        https://github.com/isayev/ASE_ANI/tree/master/ani_models/ani-1ccx_8x

    .. _Transfer Learning Paper:
        https://doi.org/10.26434/chemrxiv.6744440.v1
    """
    return BuiltinEnsemble._from_neurochem_resources('ani-1ccx_8x.info', periodic_table_index)

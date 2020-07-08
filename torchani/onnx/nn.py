import torch
from torch import Tensor
from typing import Tuple, Optional
import torchani.nn
import torchani.utils
from torchani.nn import SpeciesEnergies

class EnergyShifterOnyx(torchani.utils.EnergyShifter):

    def __init__(self, self_energies=None, fit_intercept=False):
        torch.nn.Module.__init__(self)

        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        self.register_buffer('fit_intercept', torch.tensor(fit_intercept, dtype=torch.bool))
        self.register_buffer('self_energies', self_energies)
        self.register_buffer('dummy_atom_self_energy', torch.tensor(0.0))

    def sae(self, species: Tensor) -> Tensor:
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        assert self.self_energies is not None

        self_energies = self.self_energies[species]

        # -1 is the index that determines dummy atoms
        mask = (species == -1)
        self_energies = self_energies.masked_fill(mask, self.dummy_atom_self_energy)

        if self.fit_intercept:
            return self_energies.sum(dim=1) + self.self_energies[-1]
        return self_energies.sum(dim=1)


class ANIModelOnyx(torchani.nn.ANIModel):

    def __init__(self, modules):
        torch.nn.ModuleDict.__init__(self, self.ensureOrderedDict(modules))
        # dummy buffer tensor to set devices and dtypes of dynamically created
        # float32/float64 tensors, which is necessary for onnx support, since
        # onnx.export doesn't support other.dtype / other.device when "other"
        # is not a buffer
        self.register_buffer('current_float', torch.tensor(0.0))

    def forward(self, species_aev: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        species_ = species.flatten().to(torch.long)
        aev = aev.flatten(0, 1)

        output = torch.zeros(species_.shape, device=self.current_float.device, dtype=self.current_float.dtype)

        for i, (_, m) in enumerate(self.items()):
            mask = (species_ == i)
            # onnx.export doesn't support flatten() in some contexts
            midx = mask.nonzero().view(-1)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                # in-place masked scatter is interpreted wrongly by onnx.export
                output = output.masked_scatter(mask, m(input_).view(-1))
        # onnx.export does not support view_as()
        output = output.view(species.size())
        return SpeciesEnergies(species, torch.sum(output, dim=1))


class EnsembleOnyx(torchani.nn.Ensemble):

    def __init__(self, modules):
        torch.nn.ModuleList.__init__(self, modules)
        # size has to be explicitly registered as floating point to avoid
        # onnx.export interpreting it as an int
        self.register_buffer('size', torch.tensor(float(len(modules))))

        # dummy buffer tensor to set devices and dtypes of dynamically created
        # float32/float64 tensors, which is necessary for onnx support, since
        # onnx.export doesn't support other.dtype / other.device when "other"
        # is not a buffer
        self.register_buffer('current_float', torch.tensor(0.0))

    def forward(self, species_input: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:

        species, input_ = species_input
        num_conformations = species.shape[0]
        average = torch.zeros(num_conformations, dtype=self.current_float.dtype, device=self.current_float.device)

        for x in self:
            average += x((species, input_))[1]

        average = average / self.size
        return SpeciesEnergies(species, average)

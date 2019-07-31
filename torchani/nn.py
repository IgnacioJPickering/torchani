import torch
from . import utils


class ANIModel(torch.nn.ModuleList):
    """ANI model that compute properties from species and AEVs.

    Different atom types might have different modules, when computing
    properties, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular properties.

    Arguments:
        modules (:class:`list` of :class:`torch.nn.Module`): Sequence of modules
            for each atom type. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``.  Different atom types can share a
            module by putting the same reference in :attr:`modules`.
        reducer (:class:`collections.abc.Callable`): The callable that reduce
            atomic outputs into molecular outputs. It must have signature
            `(tensor, dim) -> tensor`.
        padding_fill (float): The value to fill output of padding atoms.
            Padding values will participate in reducing, so this value should
            be appropriately chosen so that it has no effect on the result. For
            example, if the reducer is :func:`torch.sum`, then
            :attr:`padding_fill` should be 0, and if the reducer is
            :func:`torch.min`, then :attr:`padding_fill` should be
            :obj:`math.inf`.
    """

    def __init__(self, modules, reducer=torch.sum, padding_fill=0):
        super(ANIModel, self).__init__(modules)
        self.reducer = reducer
        self.padding_fill = padding_fill

    def forward(self, species_aev):
        """Forward method for the module

        This method is automatically called when an instance of the class is
        called as a function. for example:

        .. code-block:: python

            animodel = ANIModel([c_network, h_network, o_network, n_network])
            # call animodel.forward(species_input)
            species, output = animodel(species_aev)

        This usage is common for all torch forward methods. The energies output
        are the reduction of the energies calculated for each atom (by default
        the sum)
            
        Arguments:
            species_aev (:class:`tuple` of :class:`torch.Tensor`): species,
                shape ``(C, A)`` and aev, shape ``(C, A, ?)``, where ``?`` is
                the length of the AEV (in principle it could be anything; it is
                fixed for a given NNP)
        Returns:
            :class:`tuple` of :class:`torch.Tensor`: species, unchanged, and energies, shape ``(C,)``.
        """
        species, aev = species_aev
        species_ = species.flatten()
        present_species = utils.present_species(species)
        aev = aev.flatten(0, 1)

        output = torch.full_like(species_, self.padding_fill,
                                 dtype=aev.dtype)
        for i in present_species:
            mask = (species_ == i)
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            output.masked_scatter_(mask, self[i](input_).squeeze())
        output = output.view_as(species)
        return species, self.reducer(output, dim=1)


class Ensemble(torch.nn.ModuleList):
    """Compute the average output of an ensemble of modules.

    This container module holds a number of ANIModel modules and 
    reduces their combined output by averaging their results. 
    Methods ``append``, ``extend`` and ``insert`` can be called to modify 
    the modules in the list after initialization, they work exactly the 
    same as the usual python :class:`list` methods.
    
    Arguments:
        modules (:class:`list` of :class:`torch.nn.Module`): An iterable of
            modules held inside the ensemble container. By default it is
            initialized as empty.
    """

    def forward(self, species_input):
        """Forward method for the module

        Calls the forward method of each module in the class for the input
        argument given and outputs the average. 'species' goes through
        unchanged.

        This method is automatically called when an instance of the class is
        called as a function. For example:

        .. code-block:: python

            ensemble = Ensemble([animodel_1, animodel_2, animodel_3])
            # call ensemble.forward(species_input)
            species, output = ensemble(species_input)

        This usage is common for all torch forward methods. The outputs
        average all ANIModel instance outputs (after their respective
        reductions).

        Arguments:
            species_input (:class:`tuple` of :class:`torch.Tensor`): tuple of
                species, shape ``(C, A)`` and input, shape ``(C, A, ?)``.  In
                general, input will be a minibatch of species_aev, but this is
                not necessary.
        Returns:
            :class:`tuple` of :class:`torch.Tensor`: species, unchanged, and mean-energies, shape ``(C,)``.
        """
        outputs = [module(species_input)[1] for module in self]
        species, _ = species_input
        return species, sum(outputs) / len(outputs)


class Gaussian(torch.nn.Module):
    """Gaussian activation function"""
    def forward(self, x):
        """Calculates the gaussian activation function

        This is a custom gaussian-like activation layer, which has the
        funcitonal form :math:`\exp(-x \circ x)`, where :math:`\circ` is 
        the entrywise (Hadamard) product of the input with itself.


        Arguments:
            x (:class:`torch.Tensor`): input tensor to the activation layer
        Returns:
            :class:`torch.Tensor`: output tensor to the activation layer
        """
        return torch.exp(- x * x)

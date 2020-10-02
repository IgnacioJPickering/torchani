# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets

The `torchani.data.load(path)` creates an iterable of raw data,
where species are strings, and coordinates are numpy ndarrays.

You can transform these iterable by using transformations.
To do transformation, just do `it.transformation_name()`.

Available transformations are listed below:

- `species_to_indices` accepts two different kinds of arguments. It converts
    species from elements (e. g. "H", "C", "Cl", etc) into internal torchani
    indices (as returned by :class:`torchani.utils.ChemicalSymbolsToInts` or
    the ``species_to_tensor`` method of a :class:`torchani.models.BuiltinModel`
    and :class:`torchani.neurochem.Constants`), if its argument is an iterable
    of species. By default species_to_indices behaves this way, with an
    argument of ``('H', 'C', 'N', 'O', 'F', 'S', 'Cl')``  However, if its
    argument is the string "periodic_table", then elements are converted into
    atomic numbers ("periodic table indices") instead. This last option is
    meant to be used when training networks that already perform a forward pass
    of :class:`torchani.nn.SpeciesConverter` on their inputs in order to
    convert elements to internal indices, before processing the coordinates.

- `subtract_self_energies` subtracts self energies from all molecules of the
    dataset. It accepts two different kinds of arguments: You can pass a dict
    of self energies, in which case self energies are directly subtracted
    according to the key-value pairs, or a
    :class:`torchani.utils.EnergyShifter`, in which case the self energies are
    calculated by linear regression and stored inside the class in the order
    specified by species_order. By default the function orders by atomic
    number if no extra argument is provided, but a specific order may be requested.

- `remove_outliers`
- `shuffle`
- `cache` cache the result of previous transformations.
- `collate` pad the dataset, convert it to tensor, and stack them
    together to get a batch. `collate` uses a default padding dictionary
    ``{'species': -1, 'coordinates': 0.0, 'forces': 0.0, 'energies': 0.0}`` for
    padding, but a custom padding dictionary can be passed as an optional
    parameter, which overrides this default padding.

- `pin_memory` copy the tensor to pinned memory so that later transfer
    to cuda could be faster.

Note that orderings used in :class:`torchani.utils.ChemicalSymbolsToInts` and
:class:`torchani.nn.SpeciesConverter` should be consistent with orderings used
in `species_to_indices` and `subtract_self_energies`. To prevent confusion it
is recommended that arguments to intialize converters and arguments to these
functions all order elements *by their atomic number* (e. g. if you are working
with hydrogen, nitrogen and bromine always use ['H', 'N', 'Br'] and never ['N',
'H', 'Br'] or other variations).  It is possible to specify a different custom
ordering, mainly due to backwards compatibility and to fully custom atom types,
but doing so is NOT recommended, since it is very error prone.

you can also use `split` to split the iterable to pieces. use `split` as:

.. code-block:: python

    it.split(ratio1, ratio2, None)

where the None in the end indicate that we want to use all of the the rest

Example:

.. code-block:: python

    energy_shifter = torchani.utils.EnergyShifter(None)
    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(int(0.8 * size), None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()

If the above approach takes too much memory for you, you can then use dataloader
with multiprocessing to achieve comparable performance with less memory usage:

.. code-block:: python

    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(0.8, None)
    training = torch.utils.data.DataLoader(list(training), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
    validation = torch.utils.data.DataLoader(list(validation), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
"""

from os.path import join, isfile, isdir
import os
from ._pyanitools import anidataloader
from .. import utils
from .. import modules
import importlib
import functools
import math
import random
from collections import Counter
import numpy
import gc

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True

PROPERTIES = ('energies',)

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0, 
    'energies_ex': 0.0
}


def collate_fn(samples, padding=None):
    if padding is None:
        padding = PADDING

    return utils.stack_with_padding(samples, padding)


class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())


class IterableAdapterWithLength(IterableAdapter):

    def __init__(self, iterable_factory, length):
        super().__init__(iterable_factory)
        self.length = length

    def __len__(self):
        return self.length


class Transformations:
    """Convert one reenterable iterable to another reenterable iterable"""

    @staticmethod
    def standarize_keys(reenterable_iterable, nonstandard_keys):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                for std, non_std in nonstandard_keys.items():
                    d[std] = d.pop(non_std)
                yield d
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)


    @staticmethod
    def species_to_indices(reenterable_iterable, species_order=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')):
        if species_order == 'periodic_table':
            species_order = utils.PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                d['species'] = numpy.array([idx[s] for s in d['species']], dtype='i8')
                yield d
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def subtract_self_energies(reenterable_iterable, self_energies=None, species_order=None, fit_intercept=False, ex_key=None):
        intercept = 0.0
        shape_inference = False
        if isinstance(self_energies, (utils.EnergyShifter, modules.EnergyShifter)):
            if isinstance(self_energies, utils.EnergyShifter):
                fit_intercept = self_energies.fit_intercept
            shape_inference = True
            shifter = self_energies
            self_energies = {}
            counts = {}
            Y = []
            if ex_key is not None:
                Yx = []
            for n, d in enumerate(reenterable_iterable):
                species = d['species']
                count = Counter()
                for s in species:
                    count[s] += 1
                for s, c in count.items():
                    if s not in counts:
                        counts[s] = [0] * n
                    counts[s].append(c)
                for s in counts:
                    if len(counts[s]) != n + 1:
                        counts[s].append(0)
                Y.append(d['energies'])
                if ex_key is not None:
                    Yx.append(d[ex_key])

            # sort based on the order in periodic table by default
            if species_order is None:
                species_order = utils.PERIODIC_TABLE

            species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))

            X = [counts[s] for s in species]
            if fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            if ex_key is not None:
                Yx = numpy.array(Yx)
                Y_total = numpy.concatenate((Y.reshape(-1, 1), Yx), axis=-1)
                sae, _, _, _ = numpy.linalg.lstsq(X, Y_total, rcond=None)
            else:
                sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            sae_ = sae
            if fit_intercept:
                assert ex_key is None
                intercept = sae[-1]
                sae_ = sae[:-1]
            for s, e in zip(species, sae_):
                if isinstance(e, numpy.ndarray):
                    e = e.tolist()
                self_energies[s] = e
            if isinstance(shifter, utils.EnergyShifter):
                shifter.__init__(sae, shifter.fit_intercept)
            else:
                if fit_intercept:
                    shifter.__init__(self_energies=sae[:-1], intercept=sae[-1])
                else:
                    shifter.__init__(self_energies=sae)
        gc.collect()

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                e = intercept
                ex = intercept
                for s in d['species']:
                    if ex_key is None:
                        e += self_energies[s]
                    else:
                        e += self_energies[s][0]
                        ex += numpy.asarray(self_energies[s][1:])
                d['energies'] -= e
                if ex_key is not None:
                    d[ex_key] = numpy.asarray(d[ex_key]) - ex
                yield d
        if shape_inference:
            return IterableAdapterWithLength(reenterable_iterable_factory, n)
        return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def remove_outliers(reenterable_iterable, threshold1=15.0, threshold2=8.0):
        assert 'subtract_self_energies', "Transformation remove_outliers can only run after subtract_self_energies"

        # pass 1: remove everything that has per-atom energy > threshold1
        def scaled_energy(x):
            num_atoms = len(x['species'])
            return abs(x['energies']) / math.sqrt(num_atoms)
        filtered = IterableAdapter(lambda: (x for x in reenterable_iterable if scaled_energy(x) < threshold1))

        # pass 2: compute those that are outside the mean by threshold2 * std
        n = 0
        mean = 0
        std = 0
        for m in filtered:
            n += 1
            mean += m['energies']
            std += m['energies'] ** 2
        mean /= n
        std = math.sqrt(std / n - mean ** 2)

        return IterableAdapter(lambda: filter(lambda x: abs(x['energies'] - mean) < threshold2 * std, filtered))

    @staticmethod
    def shuffle(reenterable_iterable):
        list_ = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(reenterable_iterable):
        ret = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        return ret

    @staticmethod
    def collate(reenterable_iterable, batch_size, padding=None):
        def reenterable_iterable_factory(padding=None):
            batch = []
            i = 0
            for d in reenterable_iterable:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield collate_fn(batch, padding)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch, padding)

        reenterable_iterable_factory = functools.partial(reenterable_iterable_factory,
                                                         padding)
        try:
            length = (len(reenterable_iterable) + batch_size - 1) // batch_size
            return IterableAdapterWithLength(reenterable_iterable_factory, length)
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def pin_memory(reenterable_iterable):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                yield {k: d[k].pin_memory() for k in d}
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)


class TransformableIterable:
    def __init__(self, wrapped_iterable, transformations=()):
        self.wrapped_iterable = wrapped_iterable
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iterable)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self.wrapped_iterable, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        length = len(self)
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(int(n * length)):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        del self_iter
        gc.collect()
        return iters

    def __len__(self):
        return len(self.wrapped_iterable)


def load(path, nonstandard_keys=None, additional_properties=()):
    r"""As an additional option, if the dataset has a nonstandard key name you
    can use these functions to rename that key into a standard key in memory,
    without altering the dataset itself. This function takes a dictionary of the
    form {standard_name : nonstandard_name}
    """
    properties = PROPERTIES + additional_properties

    coordinates_key = 'coordinates'
    species_key = 'species'

    # species and coordinates have to be standarized first, since they 
    # are used always when iterating
    if nonstandard_keys is not None:
        if 'species' in nonstandard_keys.keys():
            species_key = nonstandard_keys.pop('species')

        if 'coordinates' in nonstandard_keys.keys():
            coordinates_key = nonstandard_keys.pop('coordinates')

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                yield from h5_files(f)
        elif isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m[species_key]
            coordinates = m[coordinates_key]
            for i in range(coordinates.shape[0]):
                ret = {'species': species, 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret
    iterable = TransformableIterable(IterableAdapter(lambda: conformations()))

    # check if there is any other key to standarize
    if nonstandard_keys:
        iterable = iterable.standarize_keys(nonstandard_keys)
    return iterable


__all__ = ['load', 'collate_fn']

# -*- coding: utf-8 -*-
"""Tools for loading/running NeuroChem input files."""

import torch
import os
import bz2
import lark
import struct
import itertools
import math
import timeit
from . import _six  # noqa:F401
import collections
import sys
from ..nn import ANIModel, Ensemble, Gaussian
from ..utils import EnergyShifter, ChemicalSymbolsToInts
from ..aev import AEVComputer
from ..optim import AdamW
import warnings
import textwrap


class Constants(collections.abc.Mapping):
    """NeuroChem constants class, dictionary-like
    
    Instances of this class can be used as arguments to initialize the module
    :class:`torchani.AEVComputer`, as in: 

    .. code-block:: python

        constants = torchani.Constants('filename')
        aev_computer = torchani.AEVComputer(**constants)

    They are dictionary-like objects that hold all constants necessary to
    compute the atomic environment vectors. Individual constants can be
    accessed as:

    .. code-block:: python

        constants = torchani.Constants('filename')
        r_cutoff_radial = constants('Rcr')
        r_cutoff_angular = constants('Rca')
        eta_radial = constants('EtaR')
        eta_angular = constants('EtaA')
        zeta = constants('Zeta')
        shifts_radial = constants('ShfR')
        shifts_angular = constants('ShfA')
        shifts_angular_theta = constants('ShfZ')

    Arguments:
        filename (:class:`str`): Path to file that holds the AEV constants.

    Attributes:
        filename (:class:`str`): Path to file that holds the AEV constants.

        species (:class:`list`[:class:`str`]): List chemical symbols of the
            elements the AEV can describe.
        num_species (:class:`int`): Number of species the AEV can describe
            (length of :attr:`species`).
        species_to_tensor (:class:`torchani.ChemicalSymbolsToInts`): Callable
            instance of ChemicalSymbolsToInts. Call to convert string or
            iterable of chemical symbols to a 1D :class:`torch.Tensor` of
            `dtype=long`, holding the associated atomic numbers.

        Rcr (:class:`float`): Value of the radial cutoff radius.
        ShfR (:class:`list`[:class:`float`]): List of radial gaussian centers.
        EtaR (:class:`list`[:class:`float`]): List of radial gaussian widths
            (one element only for current builtin models).

        Rca (:class:`float`): Value of the angular cutoff radius.
        ShfA (:class:`list`[:class:`float`]): List of angular gaussian centers.
        ShfZ (:class:`list`[:class:`float`]): List of angular cosine
            displacements.
        EtaA (:class:`list`[:class:`float`]): List of angular gaussian widths
            (one element only for current builtin models).
        Zeta (:class:`list`[:class:`float`]): List of angular Zeta exponents
            (one element only for current builtin models).
    """

    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        setattr(self, name, float(value))
                    elif name in ['EtaR', 'ShfR', 'Zeta',
                                  'ShfZ', 'EtaA', 'ShfA']:
                        value = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        setattr(self, name, torch.tensor(value))
                    elif name == 'Atyp':
                        value = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = value
                except Exception:
                    raise ValueError('unable to parse const file')

        self.num_species = len(self.species)
        self.species_to_tensor = ChemicalSymbolsToInts(self.species)

    def __iter__(self):
        yield 'Rcr'
        yield 'Rca'
        yield 'EtaR'
        yield 'ShfR'
        yield 'EtaA'
        yield 'Zeta'
        yield 'ShfA'
        yield 'ShfZ'
        yield 'num_species'

    def __len__(self):
        return 8

    def __getitem__(self, item):
        return getattr(self, item)


def load_sae(filename):
    """Return an EnergyShifter module from a SAE file
    
    Returns an object of :class:`torchani.EnergyShifter` with self atomic energies taken from
    a NeuroChem-style format sae file

    Arguments:
        filename (:class:`str`): Path to the NeuroChem `*.sae` file.
    Returns:
        energy_shifter (:class:`torchani.EnergyShifter`): EnergyShifter module that
            adds shifts to molecular energies based on SAE energies.
    """
    self_energies = []
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            self_energies.append((index, value))
    self_energies = [i for _, i in sorted(self_energies)]
    return EnergyShifter(self_energies)


def _get_activation(activation_index):
    # Activation defined in:
    # https://github.com/Jussmith01/NeuroChem/blob/stable1/src-atomicnnplib/cunetwork/cuannlayer_t.cu#L920
    if activation_index == 6:
        return None
    elif activation_index == 5:  # Gaussian
        return Gaussian()
    elif activation_index == 9:  # CELU
        return torch.nn.CELU(alpha=0.1)
    else:
        raise NotImplementedError(
            'Unexpected activation {}'.format(activation_index))


def load_atomic_network(filename):
    """Returns an sequential atomic module
    
    Returns an instance of :class:`torch.nn.Sequential` with hyperparameters
    and parameters loaded NeuroChem's .nnf, .wparam and .bparam files.

    Arguments:
        filename (:class:`str`): path to the `*.nnf` file
    Returns:
        atomic_module (:class:`torch.nn.Sequential`): Torch sequential module 
            for a given atom. Many atomic modules are used to create one
            ANIModel module.
    """

    def decompress_nnf(buffer_):
        while buffer_[0] != b'='[0]:
            buffer_ = buffer_[1:]
        buffer_ = buffer_[2:]
        return bz2.decompress(buffer_)[:-1].decode('ascii').strip()

    def parse_nnf(nnf_file):
        # parse input file
        parser = lark.Lark(r'''
        identifier : CNAME

        inputsize : "inputsize" "=" INT ";"

        assign : identifier "=" value ";"

        layer : "layer" "[" assign * "]"

        atom_net : "atom_net" WORD "$" layer * "$"

        start: inputsize atom_net

        value : SIGNED_INT
              | SIGNED_FLOAT
              | "FILE" ":" FILENAME "[" INT "]"

        FILENAME : ("_"|"-"|"."|LETTER|DIGIT)+

        %import common.SIGNED_NUMBER
        %import common.LETTER
        %import common.WORD
        %import common.DIGIT
        %import common.INT
        %import common.SIGNED_INT
        %import common.SIGNED_FLOAT
        %import common.CNAME
        %import common.WS
        %ignore WS
        ''')
        tree = parser.parse(nnf_file)

        # execute parse tree
        class TreeExec(lark.Transformer):

            def identifier(self, v):
                v = v[0].value
                return v

            def value(self, v):
                if len(v) == 1:
                    v = v[0]
                    if v.type == 'FILENAME':
                        v = v.value
                    elif v.type == 'SIGNED_INT' or v.type == 'INT':
                        v = int(v.value)
                    elif v.type == 'SIGNED_FLOAT' or v.type == 'FLOAT':
                        v = float(v.value)
                    else:
                        raise ValueError('unexpected type')
                elif len(v) == 2:
                    v = self.value([v[0]]), self.value([v[1]])
                else:
                    raise ValueError('length of value can only be 1 or 2')
                return v

            def assign(self, v):
                name = v[0]
                value = v[1]
                return name, value

            def layer(self, v):
                return dict(v)

            def atom_net(self, v):
                layers = v[1:]
                return layers

            def start(self, v):
                return v[1]

        layer_setups = TreeExec().transform(tree)
        return layer_setups

    def load_param_file(linear, in_size, out_size, wfn, bfn):
        """Load `.wparam` and `.bparam` files"""
        wsize = in_size * out_size
        fw = open(wfn, 'rb')
        w = struct.unpack('{}f'.format(wsize), fw.read())
        w = torch.tensor(w).view(out_size, in_size)
        linear.weight.data = w
        fw.close()
        fb = open(bfn, 'rb')
        b = struct.unpack('{}f'.format(out_size), fb.read())
        b = torch.tensor(b).view(out_size)
        linear.bias.data = b
        fb.close()

    networ_dir = os.path.dirname(filename)

    with open(filename, 'rb') as f:
        buffer_ = f.read()
        buffer_ = decompress_nnf(buffer_)
        layer_setups = parse_nnf(buffer_)

        layers = []
        for s in layer_setups:
            # construct linear layer and load parameters
            in_size = s['blocksize']
            out_size = s['nodes']
            wfn, wsz = s['weights']
            bfn, bsz = s['biases']
            if in_size * out_size != wsz or out_size != bsz:
                raise ValueError('bad parameter shape')
            layer = torch.nn.Linear(in_size, out_size)
            wfn = os.path.join(networ_dir, wfn)
            bfn = os.path.join(networ_dir, bfn)
            load_param_file(layer, in_size, out_size, wfn, bfn)
            layers.append(layer)
            activation = _get_activation(s['activation'])
            if activation is not None:
                layers.append(activation)

        return torch.nn.Sequential(*layers)


def load_model(species, dir_):
    """Returns an  ani_model module
    
    Returns an instance of :class:`torchani.ANIModel` loaded from
    NeuroChem's network directory.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        dir_ (:class:`str`): String for directory storing network configurations.
    Returns:
        ani_model (:class:`torchani.ANIModel`): Instance of an ANIModel module,
            using the architecture found in the NeuroChem-style format `*.nnf`
            file.
    """
    models = []
    for i in species:
        filename = os.path.join(dir_, 'ANN-{}.nnf'.format(i))
        models.append(load_atomic_network(filename))
    return ANIModel(models)


def load_model_ensemble(species, prefix, count):
    """Returns an ensemble of ANIModel modules
    
    Returns an instance of :class:`torchani.Ensemble` loaded from
    NeuroChem's network directories beginning with the given prefix.

    Arguments:
        species (:class:`collections.abc.Sequence`): Sequence of strings for
            chemical symbols of each supported atom type in correct order.
        prefix (:class:`str`): Prefix of paths of directory that networks configurations
            are stored.
        count (:class:`int`): Number of models in the ensemble.
    Returns:
        ensemble (:class:`torchani.Ensemble`): Ensemble of
            :class:`torchani.ANIModel` modules.
    """
    models = []
    for i in range(count):
        network_dir = os.path.join('{}{}'.format(prefix, i), 'networks')
        models.append(load_model(species, network_dir))
    return Ensemble(models)


def hartree2kcal(x):
    return 627.509 * x


if sys.version_info[0] > 2:

    class Trainer:
        """Train with NeuroChem training configurations.

        Arguments:
            filename (str): Input file name
            device (:class:`torch.device`): device to train the model
            tqdm (bool): whether to enable tqdm
            tensorboard (str): Directory to store tensorboard log file, set to
                ``None`` to disable tensorboard.
            aev_caching (bool): Whether to use AEV caching.
            checkpoint_name (str): Name of the checkpoint file, checkpoints
                will be stored in the network directory with this file name.
        """

        def __init__(self, filename, device=torch.device('cuda'), tqdm=False,
                     tensorboard=None, aev_caching=False,
                     checkpoint_name='model.pt'):
            try:
                import ignite
                from ..ignite import Container, MSELoss, TransformedLoss, RMSEMetric, MAEMetric, MaxAEMetric
                from ..data import BatchedANIDataset  # noqa: E402
                from ..data import AEVCacheLoader  # noqa: E402
            except ImportError:
                raise RuntimeError(
                    'NeuroChem Trainer requires ignite,'
                    'please install pytorch-ignite-nightly from PYPI')

            self.ignite = ignite

            class dummy:
                pass

            self.imports = dummy()
            self.imports.Container = Container
            self.imports.MSELoss = MSELoss
            self.imports.TransformedLoss = TransformedLoss
            self.imports.RMSEMetric = RMSEMetric
            self.imports.MaxAEMetric = MaxAEMetric
            self.imports.MAEMetric = MAEMetric
            self.imports.BatchedANIDataset = BatchedANIDataset
            self.imports.AEVCacheLoader = AEVCacheLoader

            self.warned = False

            self.filename = filename
            self.device = device
            self.aev_caching = aev_caching
            self.checkpoint_name = checkpoint_name
            self.parameters = []
            if tqdm:
                import tqdm
                self.tqdm = tqdm.tqdm
            else:
                self.tqdm = None
            if tensorboard is not None:
                import torch.utils.tensorboard
                self.tensorboard = torch.utils.tensorboard.SummaryWriter(
                    log_dir=tensorboard)
                self.training_eval_every = 20
            else:
                self.tensorboard = None

            with open(filename, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    network_setup, params = self._parse_yaml(f)
                else:
                    network_setup, params = self._parse(f.read())
                self._construct(network_setup, params)

        def _parse(self, txt):
            parser = lark.Lark(r'''
            identifier : CNAME

            outer_assign : identifier "=" value
            params : outer_assign *

            inner_assign : identifier "=" value ";"
            input_size : "inputsize" "=" INT ";"

            layer : "layer" "[" inner_assign * "]"

            atom_type : WORD

            atom_net : "atom_net" atom_type "$" layer * "$"

            network_setup: "network_setup" "{" input_size atom_net * "}"

            start: params network_setup params

            value : SIGNED_INT
                | SIGNED_FLOAT
                | STRING_VALUE

            STRING_VALUE : ("_"|"-"|"."|"/"|LETTER)("_"|"-"|"."|"/"|LETTER|DIGIT)*

            %import common.SIGNED_NUMBER
            %import common.LETTER
            %import common.WORD
            %import common.DIGIT
            %import common.INT
            %import common.SIGNED_INT
            %import common.SIGNED_FLOAT
            %import common.CNAME
            %import common.WS
            %ignore WS
            %ignore /!.*/
            ''')  # noqa: E501
            tree = parser.parse(txt)

            class TreeExec(lark.Transformer):

                def identifier(self, v):
                    v = v[0].value
                    return v

                def value(self, v):
                    if len(v) == 1:
                        v = v[0]
                        if v.type == 'STRING_VALUE':
                            v = v.value
                        elif v.type == 'SIGNED_INT' or v.type == 'INT':
                            v = int(v.value)
                        elif v.type == 'SIGNED_FLOAT' or v.type == 'FLOAT':
                            v = float(v.value)
                        else:
                            raise ValueError('unexpected type')
                    else:
                        raise ValueError('length of value can only be 1 or 2')
                    return v

                def outer_assign(self, v):
                    name = v[0]
                    value = v[1]
                    return name, value

                inner_assign = outer_assign

                def params(self, v):
                    return v

                def network_setup(self, v):
                    intput_size = int(v[0])
                    atomic_nets = dict(v[1:])
                    return intput_size, atomic_nets

                def layer(self, v):
                    return dict(v)

                def atom_net(self, v):
                    atom_type = v[0]
                    layers = v[1:]
                    return atom_type, layers

                def atom_type(self, v):
                    return v[0].value

                def start(self, v):
                    network_setup = v[1]
                    del v[1]
                    return network_setup, dict(itertools.chain(*v))

                def input_size(self, v):
                    return v[0].value

            return TreeExec().transform(tree)

        def _parse_yaml(self, f):
            import yaml
            params = yaml.safe_load(f)
            network_setup = params['network_setup']
            del params['network_setup']
            network_setup = (network_setup['inputsize'],
                             network_setup['atom_net'])
            return network_setup, params

        def _construct(self, network_setup, params):
            dir_ = os.path.dirname(os.path.abspath(self.filename))

            # delete ignored params
            def del_if_exists(key):
                if key in params:
                    del params[key]

            def assert_param(key, value):
                if key in params and params[key] != value:
                    raise NotImplementedError(key + ' not supported yet')
                del params[key]

            del_if_exists('gpuid')
            del_if_exists('nkde')
            del_if_exists('fmult')
            del_if_exists('cmult')
            del_if_exists('decrate')
            del_if_exists('mu')
            assert_param('pbc', 0)
            assert_param('force', 0)
            assert_param('energy', 1)
            assert_param('moment', 'ADAM')
            assert_param('runtype', 'ANNP_CREATE_HDNN_AND_TRAIN')
            assert_param('adptlrn', 'OFF')
            assert_param('tmax', 0)
            assert_param('ntwshr', 0)

            # load parameters
            self.const_file = os.path.join(dir_, params['sflparamsfile'])
            self.consts = Constants(self.const_file)
            self.aev_computer = AEVComputer(**self.consts)
            del params['sflparamsfile']
            self.sae_file = os.path.join(dir_, params['atomEnergyFile'])
            self.shift_energy = load_sae(self.sae_file)
            del params['atomEnergyFile']
            network_dir = os.path.join(dir_, params['ntwkStoreDir'])
            if not os.path.exists(network_dir):
                os.makedirs(network_dir)
            self.model_checkpoint = os.path.join(network_dir,
                                                 self.checkpoint_name)
            del params['ntwkStoreDir']
            self.max_nonimprove = params['tolr']
            del params['tolr']
            self.init_lr = params['eta']
            del params['eta']
            self.lr_decay = params['emult']
            del params['emult']
            self.min_lr = params['tcrit']
            del params['tcrit']
            self.training_batch_size = params['tbtchsz']
            del params['tbtchsz']
            self.validation_batch_size = params['vbtchsz']
            del params['vbtchsz']
            self.nmax = params['nmax']
            del params['nmax']

            # construct networks
            input_size, network_setup = network_setup
            if input_size != self.aev_computer.aev_length:
                raise ValueError('AEV size and input size does not match')
            atomic_nets = {}
            for atom_type in network_setup:
                layers = network_setup[atom_type]
                modules = []
                i = input_size
                for layer in layers:
                    o = layer['nodes']
                    del layer['nodes']
                    if layer['type'] != 0:
                        raise ValueError('Unsupported layer type')
                    del layer['type']
                    module = torch.nn.Linear(i, o)
                    modules.append(module)
                    activation = _get_activation(layer['activation'])
                    if activation is not None:
                        modules.append(activation)
                    del layer['activation']
                    if 'l2norm' in layer:
                        if not self.warned:
                            warnings.warn(textwrap.dedent("""
                                Currently TorchANI training with weight decay can not reproduce the training
                                result of NeuroChem with the same training setup. If you really want to use
                                weight decay, consider smaller rates and and make sure you do enough validation
                                to check if you get expected result."""))
                            self.warned = True
                        if layer['l2norm'] == 1:
                            self.parameters.append({
                                'params': [module.weight],
                                'weight_decay': layer['l2valu'],
                            })
                            self.parameters.append({
                                'params': [module.bias],
                            })
                        else:
                            self.parameters.append({
                                'params': module.parameters(),
                            })
                        del layer['l2norm']
                        del layer['l2valu']
                    else:
                        self.parameters.append({
                            'params': module.parameters(),
                        })
                    if layer:
                        raise ValueError(
                            'unrecognized parameter in layer setup')
                    i = o
                atomic_nets[atom_type] = torch.nn.Sequential(*modules)
            self.model = ANIModel([atomic_nets[s]
                                   for s in self.consts.species])
            if self.aev_caching:
                self.nnp = self.model
            else:
                self.nnp = torch.nn.Sequential(self.aev_computer, self.model)
            self.container = self.imports.Container({'energies': self.nnp}).to(self.device)

            # losses
            self.mse_loss = self.imports.MSELoss('energies')
            self.exp_loss = self.imports.TransformedLoss(
                self.imports.MSELoss('energies'),
                lambda x: 0.5 * (torch.exp(2 * x) - 1))

            if params:
                raise ValueError('unrecognized parameter')

            self.global_epoch = 0
            self.global_iteration = 0
            self.best_validation_rmse = math.inf

        def evaluate(self, dataset):
            """Evaluate on given dataset to compute RMSE and MaxAE."""
            evaluator = self.ignite.engine.create_supervised_evaluator(
                self.container,
                metrics={
                    'RMSE': self.imports.RMSEMetric('energies'),
                    'MAE': self.imports.MAEMetric('energies'),
                    'MaxAE': self.imports.MaxAEMetric('energies'),
                }
            )
            evaluator.run(dataset)
            metrics = evaluator.state.metrics
            return hartree2kcal(metrics['RMSE']), hartree2kcal(metrics['MAE']), hartree2kcal(metrics['MaxAE'])

        def load_data(self, training_path, validation_path):
            """Load training and validation dataset from file.

            If AEV caching is enabled, then the arguments are path to the cache
            directory, otherwise it should be path to the dataset.
            """
            if self.aev_caching:
                self.training_set = self.imports.AEVCacheLoader(training_path)
                self.validation_set = self.imports.AEVCacheLoader(validation_path)
            else:
                self.training_set = self.imports.BatchedANIDataset(
                    training_path, self.consts.species_to_tensor,
                    self.training_batch_size, device=self.device,
                    transform=[self.shift_energy.subtract_from_dataset])
                self.validation_set = self.imports.BatchedANIDataset(
                    validation_path, self.consts.species_to_tensor,
                    self.validation_batch_size, device=self.device,
                    transform=[self.shift_energy.subtract_from_dataset])

        def run(self):
            """Run the training"""
            start = timeit.default_timer()

            def decorate(trainer):

                @trainer.on(self.ignite.engine.Events.STARTED)
                def initialize(trainer):
                    trainer.state.no_improve_count = 0
                    trainer.state.epoch += self.global_epoch
                    trainer.state.iteration += self.global_iteration

                @trainer.on(self.ignite.engine.Events.COMPLETED)
                def finalize(trainer):
                    self.global_epoch = trainer.state.epoch
                    self.global_iteration = trainer.state.iteration

                if self.nmax > 0:
                    @trainer.on(self.ignite.engine.Events.EPOCH_COMPLETED)
                    def terminate_when_nmax_reaches(trainer):
                        if trainer.state.epoch >= self.nmax:
                            trainer.terminate()

                if self.tqdm is not None:
                    @trainer.on(self.ignite.engine.Events.EPOCH_STARTED)
                    def init_tqdm(trainer):
                        trainer.state.tqdm = self.tqdm(
                            total=len(self.training_set), desc='epoch')

                    @trainer.on(self.ignite.engine.Events.ITERATION_COMPLETED)
                    def update_tqdm(trainer):
                        trainer.state.tqdm.update(1)

                    @trainer.on(self.ignite.engine.Events.EPOCH_COMPLETED)
                    def finalize_tqdm(trainer):
                        trainer.state.tqdm.close()

                @trainer.on(self.ignite.engine.Events.EPOCH_STARTED)
                def validation_and_checkpoint(trainer):
                    trainer.state.rmse, trainer.state.mae, trainer.state.maxae = \
                        self.evaluate(self.validation_set)
                    if trainer.state.rmse < self.best_validation_rmse:
                        trainer.state.no_improve_count = 0
                        self.best_validation_rmse = trainer.state.rmse
                        torch.save(self.model.state_dict(),
                                   self.model_checkpoint)
                    else:
                        trainer.state.no_improve_count += 1

                    if trainer.state.no_improve_count > self.max_nonimprove:
                        trainer.terminate()

                if self.tensorboard is not None:
                    @trainer.on(self.ignite.engine.Events.EPOCH_STARTED)
                    def log_per_epoch(trainer):
                        elapsed = round(timeit.default_timer() - start, 2)
                        epoch = trainer.state.epoch
                        self.tensorboard.add_scalar('time_vs_epoch', elapsed,
                                                    epoch)
                        self.tensorboard.add_scalar('learning_rate_vs_epoch',
                                                    lr, epoch)
                        self.tensorboard.add_scalar('validation_rmse_vs_epoch',
                                                    trainer.state.rmse, epoch)
                        self.tensorboard.add_scalar('validation_mae_vs_epoch',
                                                    trainer.state.mae, epoch)
                        self.tensorboard.add_scalar('validation_maxae_vs_epoch',
                                                    trainer.state.maxae, epoch)
                        self.tensorboard.add_scalar(
                            'best_validation_rmse_vs_epoch',
                            self.best_validation_rmse, epoch)
                        self.tensorboard.add_scalar(
                            'no_improve_count_vs_epoch',
                            trainer.state.no_improve_count, epoch)

                        # compute training RMSE, MAE and MaxAE
                        if epoch % self.training_eval_every == 1:
                            training_rmse, training_mae, training_maxae = \
                                self.evaluate(self.training_set)
                            self.tensorboard.add_scalar(
                                'training_rmse_vs_epoch', training_rmse, epoch)
                            self.tensorboard.add_scalar(
                                'training_mae_vs_epoch', training_mae, epoch)
                            self.tensorboard.add_scalar(
                                'training_mae_vs_epoch', training_maxae, epoch)

                    @trainer.on(self.ignite.engine.Events.ITERATION_COMPLETED)
                    def log_loss(trainer):
                        iteration = trainer.state.iteration
                        loss = trainer.state.output
                        self.tensorboard.add_scalar('loss_vs_iteration',
                                                    loss, iteration)

            lr = self.init_lr

            # training using mse loss first until the validation MAE decrease
            # to < 1 Hartree
            optimizer = AdamW(self.parameters, lr=lr)
            trainer = self.ignite.engine.create_supervised_trainer(
                self.container, optimizer, self.mse_loss)
            decorate(trainer)

            @trainer.on(self.ignite.engine.Events.EPOCH_STARTED)
            def terminate_if_smaller_enough(trainer):
                if trainer.state.rmse < 10.0:
                    trainer.terminate()

            trainer.run(self.training_set, max_epochs=math.inf)

            while lr > self.min_lr:
                optimizer = AdamW(self.parameters, lr=lr)
                trainer = self.ignite.engine.create_supervised_trainer(
                    self.container, optimizer, self.exp_loss)
                decorate(trainer)
                trainer.run(self.training_set, max_epochs=math.inf)
                lr *= self.lr_decay


__all__ = ['Constants', 'load_sae', 'load_model', 'load_model_ensemble', 'Trainer']

"""Contains different versions of the main ANIModel module"""
import torch
from .residual_blocks import ResidualBlock

class Normalizer(torch.nn.Module):

    def __init__(self, mean=0., std=1.):
        super().__init__()
        assert isinstance(mean, float)
        assert isinstance(std, float)
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float))

    def forward(self, x):
        return (x - self.mean) / self.std

    def extra_repr(self):
        return f'mean={self.mean}, std={self.std}'

class AtomicNetworkClassic(torch.nn.Module):
    """Classic ANI style atomic network"""
    def __init__(self,
                 dim_in,
                 dims, 
                 activation=None,
                 mean_aev=0.,
                 std_aev=1.,
                 factor=1.,
                 final_layer_bias=False,
                 other_layers_bias=True):
        super().__init__()

        # activation can be custom or CELU
        if activation is None:
            activation = torch.nn.CELU(0.1)

        # automatically insert the first dimension
        dims.insert(0, dim_in)
        dimensions = range(len(dims) - 1)
        layers = []
        for j in dimensions:
            layers.append(torch.nn.Linear(dims[j], dims[j+1], bias=other_layers_bias))
            layers.append(activation)
        # final layer is always appended
        layers.append(torch.nn.Linear(dims[-1], 1, bias=final_layer_bias))
        self.sequential = torch.nn.Sequential(*layers)

        # the inputs can be standarized if needed, with the mean and standard
        # deviation of the aev
        self.normalizer = Normalizer(mean_aev, std_aev)

        assert isinstance(factor, float)
        factor = torch.tensor(factor, dtype=torch.float)
        self.register_buffer('factor', factor)

    @classmethod
    def like_ani1x(cls, atom):
        args_for_atoms = {
                'H' : {'dim_in' : 384, 'dims': [160, 128, 96] }, 
                'C' : {'dim_in' : 384, 'dims': [144, 112, 96] }, 
                'N' : {'dim_in' : 384, 'dims': [128, 112, 96] }, 
                'O' : {'dim_in' : 384, 'dims': [128, 112, 96] }, 
                }
        return cls(**args_for_atoms[atom], final_layer_bias=True)

    @classmethod
    def like_ani1ccx(cls, atom='H'):
        #this is just a synonym 
        return cls.like_ani1x(atom)

    @classmethod
    def like_ani2x(cls, atom='H'):
        args_for_atoms = {
                'H' : {'dim_in' : 1008, 'dims': [256, 192, 160] }, 
                'C' : {'dim_in' : 1008, 'dims': [224, 192, 160] }, 
                'N' : {'dim_in' : 1008, 'dims': [192, 160, 128] }, 
                'O' : {'dim_in' : 1008, 'dims': [192, 160, 128] }, 
                'S' : {'dim_in' : 1008, 'dims': [160, 128, 96] }, 
                'F' : {'dim_in' : 1008, 'dims': [160, 128, 96] }, 
                'Cl': {'dim_in' : 1008, 'dims': [160, 128, 96] }, 
                }
        return cls(**args_for_atoms[atom], final_layer_bias=True)


    def forward(self, x):
        out = self.normalizer(x)
        out = self.sequential(x) 
        out = out * self.factor
        return out


class AtomicNetworkResidual(torch.nn.Module):
    """Custom atomic network with residual connections"""
    def __init__(self,
                 dim_in=384,
                 dim_out1=192,
                 dim_out2=96,
                 celu_alpha=0.1,
                 batch_norm=False,
                 factor=1.):
        super().__init__()
        assert isinstance(factor, float)
        factor = torch.tensor(factor, dtype=torch.float) 
        self.register_buffer('factor', factor)

        self.residual1 = ResidualBlock(dim_in,
                                       dim_out=dim_out1,
                                       celu_alpha=celu_alpha,
                                       batch_norm=batch_norm)
        self.residual2 = ResidualBlock(dim_out1,
                                       dim_out=dim_out2,
                                       celu_alpha=celu_alpha,
                                       batch_norm=batch_norm)
        self.linear_output = torch.nn.Linear(dim_out2, 1, bias=False)

    def forward(self, x):
        out = self.residual1(x)
        out = self.residual2(out)
        return self.linear_output(out) * self.factor


class AtomicNetworkResidualMultiple(torch.nn.Module):
    """Custom residual atomic network that supports multiple outputs

    Basically the inputs go through the first two residual modules and each of
    them is multiplied by a matrix to get a different output, that part of the
    model is the "specialized" part for each output, and the residual blocks are
    common for all outputs
    """
    def __init__(self,
                 dim_in=384,
                 dim_out1=192,
                 dim_out2=96,
                 celu_alpha=0.1,
                 number_outputs=1,
                 batch_norm=False,
                 factors=None):
        super().__init__()
        if factors is None:
            factors = torch.tensor([1.] * number_outputs)
        else:
            factors = torch.tensor(factors)
            assert len(factors) == number_outputs
        self.register_buffer('factors', factors)

        # there should be one factor per applied layer
        self.residual1 = ResidualBlock(dim_in,
                                       dim_out1,
                                       celu_alpha=celu_alpha,
                                       batch_norm=batch_norm)
        self.residual2 = ResidualBlock(dim_out1,
                                       dim_out2,
                                       celu_alpha=celu_alpha,
                                       batch_norm=batch_norm)
        self.outputs_list = torch.nn.ModuleList([
            torch.nn.Linear(dim_out2, 1, bias=False)
            for _ in range(number_outputs)
        ])

    def forward(self, x):
        out = self.residual1(x)
        out = self.residual2(out)
        outputs = [m(out) * f for f, m in zip(self.factors, self.outputs_list)]
        outputs = torch.cat(outputs, dim=-1)
        return outputs


class AtomicNetworkFlexMultiple(torch.nn.Module):
    """Custom flexible atomic network with multiple residual layers"""
    def __init__(self,
                 dims=[384, 192, 96],
                 celu_alpha=0.1,
                 number_outputs=1,
                 batch_norm=False,
                 factors=None):
        super().__init__()

        # get factors or ones
        if factors is None:
            factors = torch.tensor([1.] * number_outputs)
        else:
            factors = torch.tensor(factors)
        self.register_buffer('factors', factors)
        assert len(self.factors) == number_outputs

        # basically there is a series of residual modules that are applied
        # sequentially, followed with a series of outputs which are applied
        # afterwards to get multiple outputs "in parallel" (but they are not
        # parallelized)

        residuals_shared = (
            ResidualBlock(dims[j],
                          dims[j + 1],
                          celu_alpha=celu_alpha,
                          batch_norm=batch_norm) for j in range(len(dims) - 1)
            )
        self.residuals_shared = torch.nn.Sequential(*residuals_shared)

        distinct = [
            torch.nn.Linear(dims[-1], 1, bias=False)
            for _ in range(number_outputs)
        ]
        self.distinct = torch.nn.Modulelist(distinct)


    def forward(self, x):
        # sequentially apply the residual blocks
        out = self.residuals_shared(x)
        # apply distinct non shared weights in a parallelizable loop
        outputs = [m(out) * f for m, f in zip(self.distinct, self.factors)]
        outputs = torch.cat(outputs, dim=-1)
        return outputs


class AtomicNetworkSpecFlexMultiple(torch.nn.Module):
    """custom atomic network with residual connections, many specific features"""

    # basically this is the one I want to use, the other ones are way too simple
    def __init__(self,
                 dim_in=384, 
                 dims_shared=[192, 96], # 2 shared layers, these are the OUTPUT dimensions
                 dims_specific=[96, 96],  # 2 specific layers, these are the OUTPUT dimensions
                 celu_alpha=0.1,
                 number_outputs=1,
                 batch_norm=False,
                 factors=None):
        super().__init__()

        # make dimensions match automatically, insert the input dimension as 
        # first dimension of the shared layers, and the last dimension of the 
        # shared layers as the first dimension of the specific layers
        dims_shared.insert(0, dim_in)
        if dims_specific:
            dims_specific.insert(0, dims_shared[-1])


        # if dims_specific is empty then the specific layer is just one linear
        # layer, which amounts to vector dot product, or multiplying all
        # outputs by one big matrix

        # get factors or ones
        if factors is None:
            factors = torch.tensor([1.] * number_outputs, dtype=torch.float)
        else:
            factors = torch.tensor(factors)
        self.register_buffer('factors', factors)
        assert len(self.factors) == number_outputs

        range_shared = range(len(dims_shared) - 1)
        range_specific = range(len(dims_specific) - 1)

        # shared layers get put into a Sequential module
        residuals_shared = (
            ResidualBlock(dims_shared[j],
                          dims_shared[j + 1],
                          celu_alpha=celu_alpha,
                          batch_norm=batch_norm) for j in range_shared)
        self.residuals_shared = torch.nn.Sequential(*residuals_shared)

        # specific layers
        output_modules = []
        for j in range(number_outputs):
            if dims_specific:
                residuals_specific = [
                        ResidualBlock(dims_specific[j],
                                      dims_specific[j + 1],
                                      batch_norm=batch_norm) for j in range_specific
                        ]
                residuals_specific.append(torch.nn.Linear(dims_specific[-1], 1, bias=False))
            else:
                residuals_specific = [torch.nn.Linear(dims_shared[-1], 1, bias=False)]
            # the last one is always a linear layer that ends on 1

            # turn into a sequential and append to output modules
            residuals_specific = torch.nn.Sequential(*residuals_specific)
            output_modules.append(residuals_specific)

        # turn output_modules into a correctly registered list of modules
        # to be applied in a loop
        self.output_modules = torch.nn.ModuleList(output_modules)

    def forward(self, x):
        # first go through shared modules
        out = self.residuals_shared(x)
        # afterwards go through energy-specific modules
        outputs = [
            m(out) * f for m, f in zip(self.output_modules, self.factors)
        ]
        # combine all outputs into one
        outputs = torch.cat(outputs, dim=-1)
        return outputs
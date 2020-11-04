"""Contains different versions of the main ANIModel module"""
import torch
import math

class ResidualBlock(torch.nn.Module):
    # residual block, can implement residual connection where the dimensions of both linear layers are equal
    # implements out = a( W2 * a(W1 * x + b1)) + x + b2), or
    # out = a( W2 * a(W1 * x + b1)) + W3 * x + b2), depending on wether dim_out
    # is equal to None or is some dimension, if it is equal to None it is taken to 
    # be the same as dim_in
    # can implement batch normalization after activations if wanted
    def __init__(self, dim_in, dim_out=None, celu_alpha=0.1, batch_norm=False, batch_norm_momentum=0.1, batch_norm_eps=1e-5, silu=False, gelu=False):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in
            self.linear3 = torch.nn.Identity()
        else:
            # case where dimensions are different
            self.linear3 = torch.nn.Linear(dim_in, dim_out, bias=False)

        self.linear1 = torch.nn.Linear(dim_in, dim_in)
        self.linear2 = torch.nn.Linear(dim_in, dim_out)
        if silu:
            self.activation = torch.nn.SiLU()
        elif gelu:
            self.activation = torch.nn.GELU()
        self.activation = torch.nn.CELU(alpha=celu_alpha)

        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(dim_in, momentum=batch_norm_momentum, eps=batch_norm_eps)
            self.bn2 = torch.nn.BatchNorm1d(dim_out, momentum=batch_norm_momentum, eps=batch_norm_eps)
        else:
            self.bn1 = torch.nn.Identity()
            self.bn2 = torch.nn.Identity()

    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.bn1(out)
        out = self.activation(self.linear2(out) + self.linear3(x))
        out = self.bn2(out)
        return out

class FixupBlock(torch.nn.Module):
    # residual block, can implement residual connection where the dimensions of both linear layers are equal
    # implements out = a( W2 * a(W1 * x + b1)) + x + b2), or
    # out = a( W2 * a(W1 * x + b1)) + W3 * x + b2), depending on wether dim_out
    # is equal to None or is some dimension, if it is equal to None it is taken to 
    # be the same as dim_in
    # can implement batch normalization after activations if wanted
    def __init__(self, dim_in, dim_out=None, celu_alpha=0.1, L=5):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in
            self.match = torch.nn.Identity()
        else:
            # case where dimensions are different
            self.match = torch.nn.Linear(dim_in, dim_out, bias=False)
        
        self.fixup_bias1 = torch.nn.Parameter(torch.zeros(1))
        self.fixup_bias2 = torch.nn.Parameter(torch.zeros(1))
        self.fixup_bias3 = torch.nn.Parameter(torch.zeros(1))
        self.fixup_bias4 = torch.nn.Parameter(torch.zeros(1))
        self.fixup_multiplier = torch.nn.Parameter(torch.ones(1))
        self.scale = 1/math.sqrt(L)
        
        # linear1 is initialized normally
        self.linear1 = torch.nn.Linear(dim_in, dim_in)

        # linear 2 has to be initialized with zeroes
        self.linear2 = torch.nn.Linear(dim_in, dim_out)
        self.activation = torch.nn.CELU(alpha=celu_alpha)

    def forward(self, x):
        out = self.activation(self.linear1(x + self.fixup_bias1) +
                self.fixup_bias2) 

        out = self.activation(self.fixup_multiplier * self.linear2(out +
            self.fixup_bias3) + self.fixup_bias4 + self.match(x))
        return out

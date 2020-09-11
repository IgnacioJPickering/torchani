"""Contains different versions of the main ANIModel module"""
import torch

class ResidualBlock(torch.nn.Module):
    # residual block, can implement residual connection where the dimensions of both linear layers are equal
    # implements out = a( W2 * a(W1 * x + b1)) + x + b2), or
    # out = a( W2 * a(W1 * x + b1)) + W3 * x + b2), depending on wether dim_out
    # is equal to None or is some dimension, if it is equal to None it is taken to 
    # be the same as dim_in
    # can implement batch normalization after activations if wanted
    def __init__(self, dim_in, dim_out=None, celu_alpha=0.1, batch_norm=False, batch_norm_momentum=0.01, batch_norm_eps=1e-8):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in
            self.linear3 = torch.nn.Identity()
        else:
            # case where dimensions are different
            self.linear3 = torch.nn.Linear(dim_in, dim_out, bias=False)

        self.linear1 = torch.nn.Linear(dim_in, dim_in)
        self.linear2 = torch.nn.Linear(dim_in, dim_out)
        self.activation = torch.nn.CELU(alpha=celu_alpha)

        if batch_norm:
            self.bn1 = torch.nn.BatchNorm1d(dim_in, momentum=batch_norm_momentum, eps=batch_norm_eps)
            self.bn2 = torch.nn.BatchNorm1d(dim_in, momentum=batch_norm_momentum, eps=batch_norm_eps)
        else:
            self.bn1 = torch.nn.Identity()
            self.bn2 = torch.nn.Identity()

    def forward(self, x):
        identity = x
        out = self.activation(self.linear1(x))
        out = self.bn1(out)
        out = self.activation(self.linear2(out) + identity)
        out = self.bn2(out)
        return out


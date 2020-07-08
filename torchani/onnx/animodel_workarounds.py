import torch
import math
from torch import Tensor
# workarounds for operations unsupported in onnx opset 11.  these functions
# reproduce the needed behavior of aten operations by using operations in
# opset 11 only


class Opset11Linear(torch.nn.Module):
    # essentially copied from torch.nn.Linear but simplified to avoid
    # possibility of bias being None which screws up graph, and
    # only accepts 2D inputs
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features: int, out_features: int, bias=True):
        assert bias, "This implementation of opset11 linear only accepts biased linear transforms"
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.reset_paramters()

    def reset_paramters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_):
        assert input_.dim() == 2, "This implementation of opset11 Linear only accepts 2 dimensional inputs"
        return torch.addmm(self.bias, input_, self.weight.t())

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, True)


class Opset11CELU(torch.nn.Module):
    __constants__ = ['alpha']

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        x = x / self.alpha
        return torch.nn.functional.elu(x, self.alpha)

    def extra_repr(self) -> str:
        return 'alpha={}'.format(self.alpha)

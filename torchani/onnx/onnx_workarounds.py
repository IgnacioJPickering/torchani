import torch
from torch import Tensor


# workarounds for operations unsupported in onnx opset 11.  these functions
# reproduce the needed behavior of aten operations by using operations in
# opset 11 only
def opset11_index_add():
    pass


def opset11_unique_consecutive():
    # unique consecutive and unique are very similar, with unique being
    # slower since it deletes non consecutive repetititons, and has to
    # sort in order to do that. Since in AEVComputer stuff is already
    # sorted when unique_consecutive is called, we can call unique
    # with some extra overhead without issue
    pass


def opset11_any(x: Tensor) -> Tensor:
    x = x.flatten()  # it is not necessary to know about dimensions for this
    x = (torch.sum(x) > 0)
    return x


def opset11_repeat_interleave(times_to_repeat: Tensor,
                              sequence: bool = False) -> Tensor:
    # TODO: sequence is a convenience overload that instead of repeating
    # the numbers 0 1 2 3 as 000 111 22 3 etc
    # grabs the sequence 0 1 2 3 and repeats 0123 123 23 3 this is basically
    # only done so that triu_indices can be used this is a hack and confusing,
    # and should be explained better / implemented better

    # only single argument overload is needed for AEVComputer
    assert len(times_to_repeat.shape
               ) == 1, 'opset11_repeat_interleave only accepts 1D tensors'

    # empty input means output empty tensor
    if times_to_repeat.numel() == 0:
        return times_to_repeat

    max_times_to_repeat = times_to_repeat.max()
    numbers_to_repeat = torch.arange(len(times_to_repeat),
                                     device=times_to_repeat.device,
                                     dtype=torch.long)
    # First repeat numbers_to_repeat by replicating each number to the maximum
    # value later select the ones wanted according to a mask
    numbers_to_repeat = numbers_to_repeat.repeat(max_times_to_repeat, 1)

    # Now build a mask to select only the number of repetitions necessary for
    # each row
    if sequence:
        # this is a convenience overload to be able to calculate triu_indices
        numbers_to_repeat = numbers_to_repeat.flatten()
        repeats_until_maximum = torch.arange(max_times_to_repeat,
                                             0,
                                             -1,
                                             device=times_to_repeat.device,
                                             dtype=torch.long)
        mask = (repeats_until_maximum <= times_to_repeat.view(-1, 1))
    else:
        numbers_to_repeat = numbers_to_repeat.t().flatten()
        repeats_until_maximum = torch.arange(max_times_to_repeat,
                                             device=times_to_repeat.device,
                                             dtype=torch.long)
        mask = (repeats_until_maximum < times_to_repeat.view(-1, 1))

    repeated_tensor = numbers_to_repeat.masked_select(mask.flatten())
    return repeated_tensor


def opset11_triu_indices(size):
    # triu indices using a fixed offset of 1, and with a fixed
    # row / cols of size x size
    times_to_repeat_upper = torch.arange(size - 1, 0, -1)
    upper = opset11_repeat_interleave(times_to_repeat_upper)
    lower = opset11_repeat_interleave(times_to_repeat_upper, sequence=True) + 1
    indices = torch.cat((upper.unsqueeze(0), lower.unsqueeze(0)), dim=0)
    return indices


class Opset11CELU(torch.nn.Module):
    __constants__ = ['alpha']

    def __init__(self, alpha: float = 1.0):
        super(Opset11CELU, self).__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.elu(x / self.alpha, self.alpha)

    def extra_repr(self) -> str:
        return 'alpha={}'.format(self.alpha)

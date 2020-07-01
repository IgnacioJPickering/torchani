import torch
Tensor = torch.Tensor
# workarounds for operations unsupported in onnx opset 11.  these functions
# reproduce the needed behavior of aten operations by using operations in
# opset 11 only

def triu_indices_offset_1():
    # triu indices using a fixed offset of 1
    pass

def torch_any(x : Tensor):
    x = x.flatten() # it is not necessary to know about dimensions for this
    x = (torch.sum(x) > 0)
    return x

def index_add():
    pass

def repeat_interleave_single_argument(times_to_repeat: Tensor):
    # only single argument overload is needed for AEVComputer
    assert len(times_to_repeat.shape) == 1, 'repeat_interleave_single_argument only accepts 1D tensors'
    
    max_times_to_repeat = times_to_repeat.max()
    numbers_to_repeat = torch.arange(len(times_to_repeat), device=times_to_repeat.device, dtype=torch.long)
    # First repeat numbers_to_repeat by replicating each number to the maximum
    # value later select the ones wanted according to a mask
    numbers_to_repeat = numbers_to_repeat.repeat(max_times_to_repeat, 1).t().flatten()

    # Now build a mask to select only the number of repetitions necessary for
    # each row
    repeats_until_maximum = torch.arange(max_times_to_repeat, device=times_to_repeat.device, dtype=torch.long)
    mask = (repeats_until_maximum < times_to_repeat.view(-1, 1))

    repeated_tensor = numbers_to_repeat.masked_select(mask.flatten())
    return repeated_tensor

def unique_consecutive():
    pass

def celu():
    pass

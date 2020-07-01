import torch
Tensor = torch.Tensor
# workarounds for operations unsupported in onnx opset 11.  these functions
# reproduce the needed behavior of aten operations by using operations in
# opset 11 only

def triu_indices_offset_1():
    # triu indices using a fixed offset of 1
    pass

def torch_any(x : Tensor):
    # only works for a single dimension tensor
    assert len(x.shape) == 1
    x = (torch.sum(x) > 0)
    return x

def index_add():
    pass

def repeat_interleave_single_argument():
    # only single argument overload is needed
    pass

def unique_consecutive():
    pass

def celu():
    pass


if __name__ == '__main__':
    bool_tensor = torch.tensor([True, False, False], dtype=torch.bool)
    bool_tensor = torch.tensor([False, False, False], dtype=torch.bool)
    a = torch_any(bool_tensor)
    print(a)

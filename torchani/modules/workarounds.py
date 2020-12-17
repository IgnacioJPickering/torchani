from typing import Tuple
#from collections import namedtuple

import torch
from torch.nn import functional
from torch import Tensor

def stable_argsort(input_: Tensor) -> Tensor:
    # argsort is NOT stable, it doesn't preserve order of equal elements
    # this means that it is not possible to use argsort again to recover a 
    # mapping that will preserve the original order of the elements.

    # hack to ensure stable sorting, works in pytorch 1.6.0 but may break in
    # the future, ideally stable sorting should be used here if the length of
    # the array is 2050 or more pytorch uses stable sort, so I pad the array
    # with very large numbers to go over that length if the array is small,
    # then sort and then unpad. There should not be much overhead
    if len(input_) <= 2050: 
        padded_input = functional.pad(input_, (0, 2050),'constant', 9000000000000.)
        sorted_ = torch.argsort(padded_input)[:len(input_)]
    else:
        sorted_ = torch.argsort(input_)
    return sorted_

def stable_sort(input_: Tensor) -> Tuple[Tensor, Tensor]:
    # sort is NOT stable, it doesn't preserve order of equal elements
    # this means that it is not possible to use argsort again to recover a 
    # mapping that will preserve the original order of the elements.

    # hack to ensure stable sorting, works in pytorch 1.6.0 but may break in
    # the future, ideally stable sorting should be used here if the length of
    # the array is 2050 or more pytorch uses stable sort, so I pad the array
    # with very large numbers to go over that length if the array is small,
    # then sort and then unpad. There should not be much overhead
    # TODO: JIT doesn't support namedtuple, so I will always pad no matter what
    #if len(input_) <= 2050: 
    #    padded_input = functional.pad(input_, (0, 2050),'constant', 9000000000000.)
    #    sorted_ = torch.sort(padded_input)
    #    IdxVal = namedtuple('IdxVal', 'indices values')
    #    sorted_ = IdxVal(indices=sorted_.indices[:len(input_)], values=sorted_.values[:len(input_)])
    #else:
    #    sorted_ = torch.sort(input_)
    padded_input = functional.pad(input_, (0, 2050),'constant', 9000000000000.)
    sorted_ = torch.sort(padded_input)
    sorted_ = (sorted_.indices[:len(input_)], sorted_.values[:len(input_)])
    return sorted_

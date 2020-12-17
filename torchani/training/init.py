import torch

def init_traditional(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_all_sharp(m, scale=0.01):
    if isinstance(m, torch.nn.Linear):
        # if this is an output linear layer use a smaller scale
        # for init
        #if m.weight.shape[0] == 1:
        torch.nn.init.normal_(m.weight, mean=0.0, std=scale)
        #else:
        #    torch.nn.init.kaiming_normal_(m.weight, a=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def init_sharp_gaussian(m, scale=0.001):
    if isinstance(m, torch.nn.Linear):
        # if this is an output linear layer use a smaller scale
        # for init
        # this is a hack for now
        if m.weight.shape[0] == 1 or m.weight.shape[0] == 10:
            torch.nn.init.normal_(m.weight, mean=0.0, std=scale)
        else:
            torch.nn.init.kaiming_normal_(m.weight, a=1.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def reproducible_init_nobias(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight, 1.)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.)

def reproducible_init_bias(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight, 1.)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 1.e-5)


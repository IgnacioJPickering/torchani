"""Contains different versions of the main ANIModel module"""
import torch


class AtomicNetworkAttention(torch.nn.Module):
    # I want to "attend" differently to different parts of the AEV vector
    # and afterwards I want to join those things
    def __init__(self, dim_in, num_heads):
        super().__init__()
        #3 heads works since 384//3 == 128 and 3 * 128 == 384 is good
        #kdim=None
        #vdim=None
        self.ma1 = torch.nn.MultiheadAttention(embed_dim=dim_in,
                num_heads=num_heads)
        self.linear1 = torch.nn.Linear(dim_in, dim_in//2, bias=True)
        self.linear2 = torch.nn.Linear(dim_in//2, dim_in//2//2, bias=True)
        self.linear3 = torch.nn.Linear(dim_in//2//2, 1, bias=False)
        self.a = torch.nn.CELU(alpha=0.1)

    def forward(self, x):

        x = x.unsqueeze(0)
        x = self.ma1(x, x, x, need_weights=False)[0]
        x = x.squeeze(0)

        x = self.a(self.linear1(x))
        x = self.a(self.linear2(x))
        return self.linear3(x)

        # x will have dimensions (C, 384)


import torch
import torch.nn as nn

import dhg
import torch.nn.functional as F
from utility.hgnnp_conv1 import HGNNPConv
# from e2v_cov import ThreeLayerConvNet
from utility.parser import parse_args
args = parse_args()



class HGNNP(nn.Module):
    r"""
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(
        self,
        in_channels: int,
        # hid_channels: int,
        out_channels: int,
        use_bn: bool = False,
        drop_rate: float = 0.05,
        hid_channels = 64,

    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, out_channels, use_bn=use_bn, is_last=True)
        )
        # self.theta1 = nn.Linear(64, 64, bias=True)
        # self.theta2 = nn.Linear(64, 64, bias=True)

        # self.cov = HGNNPConv(in_channels, out_channels, use_bn=use_bn, drop_rate=drop_rate)


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", v2e_weight: torch.Tensor, e2v_weight: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if i == 0:
                X_n1, X_e1 = layer(X, hg, v2e_weight, e2v_weight)
                # item_concept = torch.cat((X, X_n), dim=1)
            if i == 1:
                X_n, X_e = layer(X_n1, hg, v2e_weight, e2v_weight)



        return X_n1, X_e1, X_n, X_e #item_concept, user_concept
        # X_n, X_e = self.cov(X, hg, v2e_weight, e2v_weight)
        #return X_n, X_e
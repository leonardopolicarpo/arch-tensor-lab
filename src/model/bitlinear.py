import torch.nn as nn
from torch import Tensor

class BitLiner(nn.Linear):
  def forward(self, x: Tensor) -> Tensor:
    weight = self.weight
    scale = weight.abs().mean()
    weight_quant = weight.sign()
    return nn.functional.linear(x, weight_quant * scale, self.bias)
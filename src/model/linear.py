import torch.nn as nn
from src.model.bitlinear import BitLinear
from src.types.precision import Precision

def make_linear(in_features: int, out_features: int, precision: Precision = "fp32") -> nn.Module:
  if precision == "fp32":
    return nn.Linear(in_features, out_features)
  elif precision == "fp16":
    return nn.Linear(in_features, out_features)
  elif precision == "int4":
    raise NotImplementedError("Int4Linear coming soon!")
  elif precision == "b1.58":
    return BitLinear(in_features, out_features)
  else:
    raise ValueError(f"Unknown precision: {precision}")
from dataclasses import dataclass
from typing import Literal
from src.types.precision import Precision

@dataclass
class ModelConfig:
  vocab_size: int = 5000
  embedding_dim: int = 256
  block_size: int = 128
  num_heads: int = 8
  num_layers: int = 6

  precision: Precision = "fp32"
  head_precision: Precision = "fp32"
  device: Literal["cpu", "apu", "cuda"] = "cpu"

  name: str = "model"
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
  vocab_size: int = 5000
  embedding_dim: int = 256
  block_size: int = 128
  num_heads: int = 8
  num_layers: int = 6

  precision: Literal["fp32", "fp16", "int4", "b1.58"] = "fp32"
  device: Literal["cpu", "apu", "cuda"] = "cpu"

  name: str = "model"
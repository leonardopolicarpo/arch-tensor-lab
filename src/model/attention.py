import torch
import torch.nn as nn

class Head(nn.Module):
  def __init__(self, head_size: int, embedding_dimension: int, block_size: int):
    super().__init__()
    
    self.key = nn.Linear(embedding_dimension, head_size, bias=False)
    self.query = nn.Linear(embedding_dimension, head_size, bias=False)
    self.value = nn.Linear(embedding_dimension, head_size, bias=False)
    
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    
    k = self.key(x)
    q = self.query(x)
    
    weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    
    weights = nn.functional.softmax(weights, dim=-1)
    
    v = self.value(x)
    out = weights @ v
    
    return out
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads: int, head_size: int, embedding_dimension: int, block_size: int):
    super().__init__()
    
    self.heads = nn.ModuleList([
      Head(head_size, embedding_dimension, block_size) for _ in range(num_heads)
    ])
    
    self.proj = nn.Linear(num_heads * head_size, embedding_dimension)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    
    out = self.proj(out)
    return out
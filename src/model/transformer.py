import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention
from src.model.linear import make_linear
from src.types.precision import Precision

class FeedForward(nn.Module):
  def __init__(self, embedding_dimension: int, precision: Precision = "fp32"):
    super().__init__()
    self.net = nn.Sequential(
      make_linear(embedding_dimension, 4 * embedding_dimension, precision),
      nn.ReLU(),
      make_linear(4 * embedding_dimension, embedding_dimension, precision)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

class Block(nn.Module):
  def __init__(self, embedding_dimension: int, num_heads: int, block_size: int, precision: Precision = "fp32"):
    super().__init__()
    head_size = embedding_dimension // num_heads
    self.attention = MultiHeadAttention(num_heads, head_size, embedding_dimension, block_size)
    self.feed_forward = FeedForward(embedding_dimension, precision)
    
    self.ln1 = nn.LayerNorm(embedding_dimension)
    self.ln2 = nn.LayerNorm(embedding_dimension)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attention(self.ln1(x))
    x = x + self.feed_forward(self.ln2(x))
    return x

class LanguageModel(nn.Module):
  def __init__(
      self,
      vocabulary_size: int,
      embedding_dimension: int = 256,
      block_size: int = 128,
      num_heads: int = 8,
      num_layers: int = 6,
      precision: Precision = "fp32"
    ) -> None:
    super().__init__()

    self.block_size = block_size

    self.token_embedding_table = nn.Embedding(
      num_embeddings=vocabulary_size,
      embedding_dim=embedding_dimension
    )

    self.position_embedding_table = nn.Embedding(
      num_embeddings=block_size,
      embedding_dim=embedding_dimension
    )

    self.blocks = nn.Sequential(*[
      Block(embedding_dimension, num_heads, block_size=block_size, precision=precision) 
      for _ in range(num_layers)
    ])

    self.ln_f = nn.LayerNorm(embedding_dimension)

    self.language_modeling_head = nn.Linear(
      in_features=embedding_dimension, 
      out_features=vocabulary_size
    )
  
  def forward(
    self,
    input_indices: torch.Tensor,
    targets: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T = input_indices.shape

    token_embedding: torch.Tensor = self.token_embedding_table(input_indices)

    position_embedding: torch.Tensor = self.position_embedding_table(torch.arange(T, device=input_indices.device))

    new_embedding = token_embedding + position_embedding

    new_embedding = self.blocks(new_embedding)
    new_embedding = self.ln_f(new_embedding)

    logits: torch.Tensor = self.language_modeling_head(new_embedding)

    loss = None

    if targets is not None:
      batch_size, sequence_length, channels = logits.shape

      logits_view: torch.Tensor = logits.view(batch_size * sequence_length, channels)
      targets_view: torch.Tensor = targets.view(batch_size * sequence_length)

      loss = nn.functional.cross_entropy(logits_view, targets_view)

    return logits, loss

  def generate(
    self, 
    input_indices: torch.Tensor, 
    max_new_tokens: int, 
    temperature: float = 1.0, 
    top_k: int | None = None
  ) -> torch.Tensor:
    for _ in range(max_new_tokens):
      input_cond = input_indices[:, -self.block_size:] 
      
      logits, _ = self(input_cond)
      
      next_token_logits: torch.Tensor = logits[:, -1, :]
      
      if temperature != 1.0 and temperature > 0.0:
        next_token_logits = next_token_logits / temperature
      
      if top_k is not None:
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
          
      probabilities: torch.Tensor = nn.functional.softmax(next_token_logits, dim=-1)
      
      next_token: torch.Tensor = torch.multinomial(probabilities, num_samples=1)
      
      input_indices = torch.cat((input_indices, next_token), dim=1)
      
    return input_indices
import torch
import torch.nn as nn

class LanguageModel(nn.Module):
  def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
    super().__init__()

    self.token_embedding_table = nn.Embedding(
      num_embeddings=vocabulary_size,
      embedding_dim=embedding_dimension
    )
  
  def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
    logits: torch.Tensor = self.token_embedding_table(input_indices)
    return logits
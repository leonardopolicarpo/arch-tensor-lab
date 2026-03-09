import torch
import torch.nn as nn

class LanguageModel(nn.Module):
  def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
    super().__init__()

    self.token_embedding_table = nn.Embedding(
      num_embeddings=vocabulary_size,
      embedding_dim=embedding_dimension
    )

    self.language_modeling_head = nn.Linear(
      in_features=embedding_dimension, 
      out_features=vocabulary_size
    )
  
  def forward(
    self,
    input_indices: torch.Tensor,
    targets: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    token_embedding: torch.Tensor = self.token_embedding_table(input_indices)

    logits: torch.Tensor = self.language_modeling_head(token_embedding)

    loss = None

    if targets is not None:
      batch_size, sequence_length, channels = logits.shape

      logits_view: torch.Tensor = logits.view(batch_size * sequence_length, channels)
      targets_view: torch.Tensor = targets.view(batch_size * sequence_length)

      loss = nn.functional.cross_entropy(logits_view, targets_view)

    return logits, loss
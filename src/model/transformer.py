import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention

class LanguageModel(nn.Module):
  def __init__(self, vocabulary_size: int, embedding_dimension: int, block_size: int) -> None:
    super().__init__()

    self.block_size = block_size

    self.token_embedding_table = nn.Embedding(
      num_embeddings=vocabulary_size,
      embedding_dim=embedding_dimension
    )

    num_heads = 4
    head_size = embedding_dimension // num_heads

    self.attention_heads = MultiHeadAttention(
      num_heads=num_heads,
      head_size=head_size,
      embedding_dimension=embedding_dimension,
      block_size=block_size
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

    context_aware_embedding: torch.Tensor = self.attention_heads(token_embedding)

    logits: torch.Tensor = self.language_modeling_head(context_aware_embedding)

    loss = None

    if targets is not None:
      batch_size, sequence_length, channels = logits.shape

      logits_view: torch.Tensor = logits.view(batch_size * sequence_length, channels)
      targets_view: torch.Tensor = targets.view(batch_size * sequence_length)

      loss = nn.functional.cross_entropy(logits_view, targets_view)

    return logits, loss
  
  def generate(self, input_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
      input_cond = input_indices[:, -self.block_size:] 
      
      logits, _ = self(input_cond)
      
      next_token_logits: torch.Tensor = logits[:, -1, :]
      probabilities: torch.Tensor = nn.functional.softmax(next_token_logits, dim=-1)
      next_token: torch.Tensor = torch.multinomial(probabilities, num_samples=1)
      
      input_indices = torch.cat((input_indices, next_token), dim=1)
      
    return input_indices
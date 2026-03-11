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
  
  def generate(self, input_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
      logits, _ = self(input_indices)
      
      next_token_logits: torch.Tensor = logits[:, -1, :]
      
      probabilities: torch.Tensor = nn.functional.softmax(next_token_logits, dim=-1)
      
      next_token: torch.Tensor = torch.multinomial(probabilities, num_samples=1)
      
      input_indices = torch.cat((input_indices, next_token), dim=1)
        
    return input_indices
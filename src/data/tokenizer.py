import os
import torch

class Tokenizer:
  def __init__(self, raw_text_path: str) -> None:
    if not os.path.exists(raw_text_path):
      raise FileNotFoundError(f"File not found: {raw_text_path}")
    
    with open(raw_text_path, 'r', encoding='utf-8') as file:
      self.text = file.read()

    self.characters = sorted(list(set(self.text)))
    self.vocabulary_size = len(self.characters)

    self.string_to_index = { character:index for index,character in enumerate(self.characters) }
    self.index_to_string = { index:character for index,character in enumerate(self.characters) }

  def encode(self, string: str) -> list[int]:
    return [self.string_to_index[character] for character in string]

  def decode(self, list_of_indices: list[int]) -> str:
    return ''.join([self.index_to_string[index] for index in list_of_indices])

  def save_data(self, output_path: str) -> None:
    tokens = self.encode(self.text)

    # long (int64) standard for indeces in embeddings
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data_tensor, output_path)

    print(f"Vocabulary: {self.vocabulary_size} characters")
    print(f"Binary file saved in: {output_path}")
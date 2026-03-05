import os

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

  def decode(self, list_of_integers: list[int]) -> str:
    pass

  def save_data(self, output_path: str) -> None:
    pass
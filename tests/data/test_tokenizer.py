import os
import pytest
import torch
from data.tokenizer import Tokenizer

@pytest.fixture
def sample_data_path(tmp_path) -> str:
  diretory = tmp_path / "raw"
  diretory.mkdir()
  file = diretory / "test_bash.txt"
  file.write_text("sudo pacman -Syu\nif [ -d ]; then\n", encoding="utf-8")
  return str(file)

def test_vocabulary_size(sample_data_path: str) -> None:
  tokenizer = Tokenizer(sample_data_path)
  assert tokenizer.vocabulary_size > 0
  assert isinstance(tokenizer.vocabulary_size, int)

def test_encode_decode_integrity(sample_data_path: str) -> None:
  tokenizer = Tokenizer(sample_data_path)
  original_text = "sudo pacman"

  encoded = tokenizer.encode(original_text)
  decoded = tokenizer.decode(encoded)

  assert original_text == decoded
  assert all(isinstance(index, int) for index in encoded)

def test_save_data_persistence(sample_data_path, tmp_path):
  tokenizer = Tokenizer(sample_data_path)
  output_path = str(tmp_path / "processed" / "data.pt")

  tokenizer.save_data(output_path)

  assert os.path.exists(output_path)
  loaded_tensor = torch.load(output_path, weights_only=True)
  assert isinstance(loaded_tensor, torch.Tensor)
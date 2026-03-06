import pytest
import torch
from data.loader import DataLoader

@pytest.fixture
def sample_tensor_path(tmp_path):
  path = tmp_path / "test_data.pt"
  data = torch.arange(100, dtype=torch.long)
  torch.save(data, path)
  return str(path)

def test_dataloader_init_split(sample_tensor_path):
  batch_size, block_size = 4, 8
  loader = DataLoader(sample_tensor_path, batch_size, block_size, split_ratio=0.8)
  
  assert len(loader.train_data) == 80
  assert len(loader.val_data) == 20

def test_dataloader_batch_shapes(sample_tensor_path):
  batch_size, block_size = 4, 8
  loader = DataLoader(sample_tensor_path, batch_size, block_size)
  x, y = loader.get_batch('train')
  
  assert x.shape == (batch_size, block_size)
  assert y.shape == (batch_size, block_size)

def test_dataloader_target_offset(sample_tensor_path):
  batch_size, block_size = 1, 10
  loader = DataLoader(sample_tensor_path, batch_size, block_size)
  x, y = loader.get_batch('train')
  
  assert torch.equal(x[0, 1:], y[0, :-1])
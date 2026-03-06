import torch

class DataLoader:
  def __init__(
    self,
    data_path: str,
    batch_size: int,
    block_size: int,
    split_ratio: float = 0.9
  ):
    self.batch_size = batch_size
    self.block_size = block_size

    data = torch.load(data_path, weights_only=True)

    n = int(split_ratio * len(data))
    self.train_data = data[:n]
    self.val_data = data[n:]

  def get_batch(self, split: str = 'train') -> tuple:
    data = self.train_data if split == 'train' else self.val_data

    start_indices = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))

    x = torch.stack([data[index:index+self.block_size] for index in start_indices])
    y = torch.stack([data[index+1:index+self.block_size+1] for index in start_indices])

    return x, y
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

  def get_batch(self, split: str = 'train') -> None:
    data = self.train_data if split == 'train' else self.val_data

    print(f"data: {data}")
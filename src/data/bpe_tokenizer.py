import os
import torch

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPETokenizer:
  def __init__(self, vocab_size=5000):
    self.vocab_size = vocab_size
    self.tokenizer = HFTokenizer(BPE(unk_token="[UNK]"))
    self.tokenizer.pre_tokenizer = Whitespace()
    self.model_path = "data/processed/bpe_vocab.json"

  def train_and_save(self, file_path):
    print(f"[*] Treinando BPE Tokenizer no arquivo: {file_path}...")
    trainer = BpeTrainer(
      special_tokens=["[UNK]", "[User]:", "[Agent]:"], 
      vocab_size=self.vocab_size
    )
    self.tokenizer.train(files=[file_path], trainer=trainer)
    self.tokenizer.save(self.model_path)
    print(f"[*] Vocabulário BPE salvo em: {self.model_path} (Tamanho: {self.tokenizer.get_vocab_size()})")

  def load(self):
    if not os.path.exists(self.model_path):
      raise FileNotFoundError(f"Arquivo {self.model_path} não encontrado. Treine primeiro.")
    self.tokenizer = HFTokenizer.from_file(self.model_path)
    self.vocab_size = self.tokenizer.get_vocab_size()

  def encode(self, text: str) -> list[int]:
    return self.tokenizer.encode(text).ids

  def decode(self, ids: list[int]) -> str:
    return self.tokenizer.decode(ids)

  def save_data(self, raw_text_path: str, output_pt_path: str):
    print(f"[*] Lendo {raw_text_path} para converter em tensores BPE...")
    with open(raw_text_path, 'r', encoding='utf-8') as f:
      text = f.read()
    
    print("[*] Codificando texto (isso pode levar uns segundos)...")
    data = torch.tensor(self.encode(text), dtype=torch.long)
    torch.save(data, output_pt_path)
    print(f"[*] Dataset BPE salvo em {output_pt_path}!")
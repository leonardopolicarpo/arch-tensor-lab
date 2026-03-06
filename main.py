import argparse
import os
from src.data.tokenizer import Tokenizer
from src.data.loader import DataLoader

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', '-f', type=str, default='bash_commands.txt')
  parser.add_argument('--target', '-t', type=str, default='bash_data.pt')

  args = parser.parse_args()

  raw_data_path = "data/raw/" + args.file
  processed_data_path = "data/processed/" + args.target

  if not os.path.exists(processed_data_path):
    print(f"[*] Generating binary in {processed_data_path}...")
    tokenizer = Tokenizer(raw_text_path=raw_data_path)
    tokenizer.save_data(processed_data_path)
  else:
    print("[!] Binary already exists. Skipping the tokenization step")

  loader = DataLoader(
    data_path=processed_data_path,
    batch_size=32,
    block_size=64
  )
  x, y = loader.get_batch(split="train")

  print(f"Batch X shape: {x.shape}")
  print(f"Batch Y shape: {y.shape}")

if __name__ == "__main__":
  main()
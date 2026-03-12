import argparse
import os
import torch
import torch.optim as optim
from src.data.tokenizer import Tokenizer
from src.data.loader import DataLoader
from src.model.transformer import LanguageModel

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', '-f', type=str, default='bash_commands.txt')
  parser.add_argument('--target', '-t', type=str, default='bash_data.pt')
  parser.add_argument('--model', '-m', type=str, default='bash_model')
  parser.add_argument('--train', action='store_true', help='Executar loop de treinamento')
  return parser.parse_args()

def generate_sample(model: LanguageModel, tokenizer: Tokenizer, max_new_tokens: int = 100) -> str:
  model.eval() 
  initial_context = torch.zeros((1, 1), dtype=torch.long)
  generated_tokens = model.generate(initial_context, max_new_tokens)
  return tokenizer.decode(generated_tokens[0].tolist())

def train_model(model: LanguageModel, loader: DataLoader, steps: int = 1000, lr: float = 1e-3) -> None:
  print(f"\n[*] --- Iniciando Treinamento ({steps} passos) ---")
  optimizer = optim.AdamW(model.parameters(), lr=lr)
  
  model.train() 
  
  for step in range(steps):
    xb, yb = loader.get_batch(split="train")
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
      print(f"Passo {step:4d} | Loss atual: {loss.item():.4f}")
          
  print("\n[*] --- Treinamento Concluído! ---")
  print(f"[*] Loss Final: {loss.item():.4f}")

def main():
  args = parse_arguments()
  
  BLOCK_SIZE = 128
  EMBED_DIM = 256
  N_HEADS = 8
  N_LAYERS = 6

  raw_data_path = f"data/raw/{args.file}"
  processed_data_path = f"data/processed/{args.target}"
  model_weights_path = f"data/processed/{args.model}_weights.pt"

  tokenizer = Tokenizer(raw_text_path=raw_data_path)
  if not os.path.exists(processed_data_path):
    print(f"[*] Generating binary in {processed_data_path}...")
    tokenizer.save_data(processed_data_path)
  
  vocabulary_size = tokenizer.vocabulary_size
  loader = DataLoader(data_path=processed_data_path, batch_size=32, block_size=BLOCK_SIZE)
  
  model = LanguageModel(
    vocabulary_size,
    embedding_dimension=EMBED_DIM,
    block_size=BLOCK_SIZE,
    num_heads=N_HEADS,
    num_layers=N_LAYERS
  )

  if os.path.exists(model_weights_path):
    print(f"\n[*] Cérebro encontrado! Carregando pesos de: {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path))
  else:
    print("\n[*] Cérebro não encontrado. Inicializando com pesos aleatórios.")

  if args.train:
    print("\n[*] Teste de Geração (Antes do treino) ---")
    print(generate_sample(model, tokenizer, max_new_tokens=100))
    
    train_model(model, loader, steps=1000)
    
    print(f"\n[*] Salvando os novos pesos em {model_weights_path}...")
    torch.save(model.state_dict(), model_weights_path)
  else:
    print("\n[*] Modo de inferência. Pulando treinamento...")

  print("\n[*] --- Teste de Geração Final ---")
  print(generate_sample(model, tokenizer, max_new_tokens=200))

if __name__ == "__main__":
  main()
import argparse
import os
import torch
import torch.optim as optim
from src.data.bpe_tokenizer import BPETokenizer
from src.data.loader import DataLoader
from src.model.transformer import LanguageModel
from src.config.model_config import ModelConfig

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file', '-f', type=str, default='chat_pro.txt')
  parser.add_argument('--target', '-t', type=str, default='agent_v4.pt')
  parser.add_argument('--model', '-m', type=str, default='agent_v4')
  parser.add_argument('--train', action='store_true', help='Executar loop de treinamento')
  parser.add_argument('--steps', '-s', type=int, default=2000, help='Quantidade de passos para treinar')
  return parser.parse_args()

def generate_sample(model: LanguageModel, tokenizer: BPETokenizer, prompt: str = "", max_new_tokens: int = 100) -> str:
  model.eval() 
  
  if prompt:
    context_list = tokenizer.encode(prompt)
    initial_context = torch.tensor([context_list], dtype=torch.long)
  else:
    initial_context = torch.zeros((1, 1), dtype=torch.long)
    
  generated_tokens = model.generate(initial_context, max_new_tokens)
  return tokenizer.decode(generated_tokens[0].tolist())

def train_model(model: LanguageModel, loader: DataLoader, weights_path: str, steps: int = 1000, lr: float = 1e-4) -> None:
  print(f"\n[*] --- Iniciando Treinamento ({steps} passos) ---")
  optimizer = optim.AdamW(model.parameters(), lr=lr)
  
  model.train() 
  
  for step in range(1, steps + 1):
    xb, yb = loader.get_batch(split="train")
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0 or step == 1:
      print(f"Passo {step:4d} | Loss atual: {loss.item():.4f}")
    
    if step % 500 == 0:
      print(f"[*] {step} passos atingidos. Salvando checkpoint em {weights_path}...")
      torch.save(model.state_dict(), weights_path)
          
  print("\n[*] --- Treinamento Concluído! ---")
  print(f"[*] Loss Final: {loss.item():.4f}")

def main():
  args = parse_arguments()

  config = ModelConfig(
    name=args.model,
    precision="fp32",
    head_precision="fp32",
    device="cpu"
  )

  raw_data_path = f"data/raw/{args.file}"
  processed_data_path = f"data/processed/{args.target}"
  model_weights_path = f"data/processed/{args.model}_weights.pt"
  vocab_path = "data/processed/bpe_vocab.json"

  tokenizer = BPETokenizer(vocab_size=config.vocab_size)
  
  if not os.path.exists(vocab_path):
    print(f"[*] Vocabulário não encontrado. Treinando BPE...")
    tokenizer.train_and_save(raw_data_path)
  
  tokenizer.load()

  if not os.path.exists(processed_data_path):
    print(f"[*] Gerando binário de tokens...")
    tokenizer.save_data(raw_data_path, processed_data_path)
  
  vocabulary_size = tokenizer.vocab_size
  loader = DataLoader(
    data_path=processed_data_path,
    batch_size=32,
    block_size=config.block_size
  )
  
  model = LanguageModel(
    vocabulary_size,
    embedding_dimension=config.embedding_dim,
    block_size=config.block_size,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    precision=config.precision,
    head_precision=config.head_precision
  )

  if os.path.exists(model_weights_path):
    print(f"\n[*] Modelo encontrado! Carregando pesos de: {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path))
  else:
    print(f"\n[*] Modelo {args.model} não encontrado. Inicializando do zero.")

  total_params = sum(p.numel() for p in model.parameters())
  print(f"[*] Tamanho do Modelo: {total_params / 1e6:.2f} Milhões de parâmetros")

  if args.train:
    print("\n[*] Teste de Geração (Antes do treino) ---")
    print(generate_sample(model, tokenizer, max_new_tokens=50))
    
    train_model(model, loader, model_weights_path, steps=args.steps)
    
    print(f"\n[*] Salvando pesos finais em {model_weights_path}...")
    torch.save(model.state_dict(), model_weights_path)
  else:
    print("\n[*] Modo de inferência. Pulando treinamento...")

  print("\n[*] --- Teste de Geração Final ---")

  prompt = "[User]: qual seu nome?\n[Agent]:"

  print(generate_sample(model, tokenizer, prompt, max_new_tokens=50))

if __name__ == "__main__":
  main()
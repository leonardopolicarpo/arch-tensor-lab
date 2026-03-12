import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.tokenizer import Tokenizer
from src.model.transformer import LanguageModel

BLOCK_SIZE = 128
EMBED_DIM = 256
N_HEADS = 8
N_LAYERS = 6

def clear_console():
  os.system('clear' if os.name == 'posix' else 'cls')

def load_agent(weights_path, raw_data_path):
  print(f"[*] Acordando o Agente ({weights_path})...")
  
  tokenizer = Tokenizer(raw_text_path=raw_data_path)
  model = LanguageModel(
    tokenizer.vocabulary_size, 
    embedding_dimension=EMBED_DIM, 
    block_size=BLOCK_SIZE, 
    num_heads=N_HEADS, 
    num_layers=N_LAYERS
  )
  
  model.load_state_dict(torch.load(weights_path))
  model.eval()
  return model, tokenizer

def responder(model, tokenizer, prompt_usuario, max_tokens=150):
  full_prompt = f"[User]: {prompt_usuario}\n[Agent]:"
  
  context = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long)
  
  generated = model.generate(context, max_new_tokens=max_tokens)
  texto_gerado = tokenizer.decode(generated[0].tolist())

  resposta_pura = texto_gerado[len(full_prompt):]
  if "[User]" in resposta_pura:
    resposta_pura = resposta_pura.split("[User]")[0]
      
  return resposta_pura.strip()

def main():
  weights = "data/processed/agent_v2_weights.pt"
  data_source = "data/raw/chat_pro.txt"

  if not os.path.exists(weights):
    print(f"[!] Erro: Pesos não encontrados em {weights}")
    return

  model, tokenizer = load_agent(weights, data_source)
  
  clear_console()
  print("---------------------------------------------------")
  print("   Arch-Tensor-Lab: CORE AGENT INTERACTIVE")
  print("---------------------------------------------------")
  print(" Digite 'sair' para encerrar ou 'clear' para limpar.\n")

  while True:
    try:
      user_input = input("\033[1;32m>> Você:\033[0m ")
      
      if user_input.lower() == 'sair':
        break
      if user_input.lower() == 'clear':
        clear_console()
        continue
      if not user_input.strip():
        continue

      print("\n\033[1;34m[*] Agente:\033[0m ", end="", flush=True)
      
      resposta = responder(model, tokenizer, user_input)
      print(f"{resposta}\n")

    except KeyboardInterrupt:
      print("\n[!] Encerrando...")
      break

if __name__ == "__main__":
  main()
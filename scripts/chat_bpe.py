import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.bpe_tokenizer import BPETokenizer
from src.model.transformer import LanguageModel

BLOCK_SIZE = 128
EMBED_DIM = 256
N_HEADS = 8
N_LAYERS = 6
VOCAB_SIZE = 5000

def clear_console():
  os.system('clear' if os.name == 'posix' else 'cls')

def load_agent(weights_path):
  print(f"[*] Acordando o Agente ({weights_path})...")
  
  tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
  tokenizer.load() 
  
  model = LanguageModel(
    tokenizer.vocab_size,
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
  
  context_ids = tokenizer.encode(full_prompt)
  context = torch.tensor([context_ids], dtype=torch.long)
  
  generated = model.generate(context, max_new_tokens=max_tokens, temperature=0.7, top_k=50)
  texto_gerado = tokenizer.decode(generated[0].tolist())

  if "[Agent]:" in texto_gerado:
    resposta_pura = texto_gerado.split("[Agent]:")[-1]
  else:
    resposta_pura = texto_gerado[len(full_prompt):]

  if "[User]:" in resposta_pura:
    resposta_pura = resposta_pura.split("[User]:")[0]
    
  return resposta_pura.strip()

def main():
  weights = "data/processed/agent_v3_weights.pt"

  if not os.path.exists(weights):
    print(f"[!] Erro: Pesos da V3 não encontrados em {weights}")
    print("[*] Certifique-se de que o treinamento terminou e salvou os pesos.")
    return

  model, tokenizer = load_agent(weights)
  
  clear_console()
  print("---------------------------------------------------")
  print("   Arch-Tensor-Lab: CORE AGENT INTERACTIVE (V3-BPE)")
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
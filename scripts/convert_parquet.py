import pyarrow.parquet as pq
import os

def convert_minimalist():
  input_path = "data/raw/alpaca.parquet"
  output_path = "data/raw/chat_pro.txt"

  if not os.path.exists(input_path):
    print(f"[!] Erro: {input_path} nao encontrado.")
    return
  
  table = pq.read_table(input_path, columns=['instruction', 'input', 'output'])
  
  data = table.to_pylist()
  
  print(f"[*] Registros encontrados: {len(data)}")
  
  with open(output_path, "w", encoding="utf-8") as f:
    for item in data[:15000]:
      instr = str(item['instruction']).strip()
      inp = str(item['input']).strip()
      out = str(item['output']).strip()

      if instr and out:
        prompt = f"[User]: {instr}"
        if inp and inp.lower() not in ["none", "nan", ""]:
          prompt += f"\nContexto: {inp}"
        
        f_out = f"{prompt}\n[Agent]: {out}\n\n"
        f.write(f_out)

  print(f"[+] Sucesso! Arquivo gerado em: {output_path}")

if __name__ == "__main__":
  convert_minimalist()
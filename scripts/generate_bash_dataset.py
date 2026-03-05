import os

BASH_SNIPPETS = [
  # Condicionais básicos
  "if [ -d \"/etc/pacman.d\" ]; then\n  echo \"Arch Linux detected\"\nfi\n",
  "if systemctl is-active --quiet i3; then\n  echo \"i3 is running\"\nelse\n  echo \"i3 is down\"\nfi\n",
  
  # Loops e iterações
  "for file in *.txt; do\n  cat \"$file\" | grep \"error\"\ndone\n",
  "while read -r line; do\n  echo \"Processing: $line\"\ndone < input.log\n",
  
  # Comandos pacman e manutenção
  "sudo pacman -Syu --noconfirm\n",
  "pacman -Qdtq | sudo pacman -Rs -\n",
  "sudo pacman -S --needed base-devel git\n",
  
  # Manipulação de texto com awk e grep
  "ps aux | awk '{print $1, $2, $11}' | grep \"python\"\n",
  "cat /var/log/syslog | grep \"CRON\" | awk -F': ' '{print $2}'\n",
  "find . -type f -name \"*.py\" -exec grep -Hn \"import\" {} \\;\n",
  
  # Funções customizadas
  "function update_system() {\n  echo \"Updating Arch...\"\n  sudo pacman -Syu\n}\n",
  "function extract() {\n  if [ -f $1 ] ; then\n    tar xzvf $1\n  else\n    echo \"File not found\"\n  fi\n}\n"
]

def generate_dataset(output_path: str, multiplier: int = 1000):
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  
  print(f"Generating dataset in: {output_path}")
  
  with open(output_path, "w", encoding="utf-8") as f:
    for _ in range(multiplier):
      for snippet in BASH_SNIPPETS:
        f.write(snippet + "\n")
              
  file_size = os.path.getsize(output_path) / (1024 * 1024)
  print(f"Success! Dataset created with {file_size:.2f} MB.")

if __name__ == "__main__":
  output_file = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "bash_commands.txt")
  generate_dataset(output_file, multiplier=5000)
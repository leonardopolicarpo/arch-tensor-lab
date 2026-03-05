# Ingestão de Dados e Tokenização

Este documento detalha o primeiro estágio do **arch-tensor-lab**, que é a criação de um dataset sintético baseado no ecossistema Arch Linux e a implementação de um Tokenizer de nível de caractere.

---

## 1. A Matéria-Prima

Para treinar uma IA a entender lógica e sintaxe, escolhi comandos do terminal Bash. Diferente da linguagem natural humana, o Bash possui uma **rigidez sintática** que facilita o aprendizado inicial da rede neural.

### Por que Bash e Arch Linux?
- **Determinismo:** Comandos como `pacman -Syu` ou `find . -type f` seguem regras claras de flags e argumentos.
- **Utilidade Prática:** O modelo aprende a "pensar" dentro do ambiente onde está sendo desenvolvido.
- **Volume Controlado:** Gerei aproximadamente **3.5MB** de texto (cerca de 5.000 snippets), o que é ideal para testes rápidos em CPU.
- **Uso Pessoal:** Utilizo o ecossistema arch no dia a dia.

---

## 2. Arquitetura do Tokenizer

Optei por um **Character-Level Tokenizer** que, em vez de palavras inteiras, a unidade mínima de processamento é o caractere individual.

### Implementação Técnica:
- **Unicidade e Determinismo:** Utilizei `set()` para extrair os caracteres únicos e `sorted()` para garantir que o índice (ID) de cada caractere seja sempre o mesmo em qualquer execução.
- **Mapeamento:** Criei dois dicionários de consulta:
  - `string_to_index`: Traduz letras/símbolos em números para a rede neural.
  - `index_to_string`: Traduz os números de volta para texto legível.
- **Eficiência:** O texto processado é convertido em um **Tensor do PyTorch** (`torch.long`) e salvo em formato binário `.pt`, garantindo carregamento ultrarrápido durante o treino.

---

## 3. Desafios de Engenharia e Soluções

Enfrentei alguns obstáculos de infraestrutura que moldaram a arquitetura do projeto:

### O Gargalo do Disco
Durante a instalação das dependências, o PyTorch tentou baixar gigabytes de drivers CUDA (Nvidia). Como o laboratório é focado em **CPU-first**, resolvi isso com:
1. `pip cache purge`: Limpeza de arquivos temporários para liberar espaço.
2. `pip install torch --index-url https://download.pytorch.org/whl/cpu`: Instalação forçada de uma versão leve, otimizada apenas para processadores.

### Estrutura de Pacotes (src-layout)
Adotei o padrão industrial `src/` para isolar o código fonte dos testes e documentação. Isso exigiu a configuração do `pyproject.toml` com `tool.setuptools.packages.find` para que os módulos fossem reconhecidos corretamente pelo ambiente virtual.

---

## 4. Garantia de Qualidade

Implementei testes unitários utilizando **Pytest** desde o início.
- **Isolamento:** Usa *fixtures* para criar datasets temporários, e garantir que os testes não dependam de arquivos externos.
- **Integridade:** Valida se o processo de `encode -> decode` retorna exatamente a string original sem perda de informação.
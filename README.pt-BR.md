# arch-tensor-lab 🚀

> 🇺🇸 **This README is available in [English](./README.md).**

Bem-vindo ao **arch-tensor-lab**. Este é meu espaço de experimentação focado em construir uma arquitetura de Large Language Model (LLM) do absoluto zero.
Meu objetivo é pedagógico e braçal: entender a matemática, a engenharia de dados e a implementação de Transformers, começando pela base.

---

## 🛑 A Filosofia do Projeto

- **From Scratch:** Implemento cada componente (Tokenização, Embeddings, Attention, Backpropagation) para consolidar o aprendizado.
- **CPU-First:** Desenvolvido em um ambiente Arch Linux otimizado para CPU (Ryzen 5 5600G). Com o objetivo de ser um guia para eficiência.
- **Estilo "Inca" de Código:** Cada linha de código é escrita manualmente. Usando assistentes de IA apenas para agilizar tarefas repetitivas.

---

## 🛠️ Status Atual

### ✅ Fase 01: Ingestão e Tokenização
- **Dataset Sintético:** ~3.5MB de comandos Bash e lógica do Arch Linux.
- **Character-Level Tokenizer:** Mapeamento determinístico de caracteres para IDs.
- **Testes Unitários:** Suite completa com `pytest`.

### ✅ Fase 02: DataLoader e Dinâmica de Batches
- **Matriz de Treino:** Implementação de lógica de fatiamento ($32 \times 64$).
- **Sorteio Semântico:** Uso de `torch.randint` com margens de segurança para evitar estouro de índice.
- **Orquestração:** Script `run.sh` e `main.py` para automação do pipeline.

---

## 📂 Estrutura do Repositório

```text
arch-tensor-lab/
├── data/           # Datasets (raw e processed)
├── docs/           # Documentação detalhada em EN/PT-BR
├── src/            # Código-fonte (Tokenizer, Loader, Model)
├── tests/          # Testes automatizados (pytest)
└── pyproject.toml  # Gestão de dependências e ambiente
```

---

## 🚀 Como Executar

Se você estiver em um ambiente Arch Linux (ou qualquer distro Linux) e quiser replicar o laboratório localmente:

1. **Clone o repositório:**
```bash
git clone https://github.com/leonardopolicarpo/arch-tensor-lab.git
cd arch-tensor-lab
```

2. **Crie o ambiente virtual e instale as dependências:**
*(Nota: O laboratório usa a versão CPU-only do PyTorch para economizar recursos de disco e GPU)*
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

3. **Rode os testes:**
```bash
pytest
```

---

## 📖 Documentação Detalhada

Confira a explicação técnica de cada etapa:

- **01: Ingestão e Tokenização** — [Português](docs/pt-br/01-data-ingestion-and-tokenization.md) | [English](docs/en/01-data-ingestion-and-tokenization.md)
- **02: DataLoader e Batches** — [Português](docs/pt-br/02-dataloader.md) | [English](docs/en/02-dataloader.md)

---

> "Entender a base é o que diferencia o engenheiro do utilizador de ferramentas."
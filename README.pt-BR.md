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

## 🛠️ Status Atual: Fase 01 - Ingestão e Tokenização

Até agora, o laboratório já conta com:
- **Dataset Sintético:** ~3.5MB de comandos Bash e lógica do ecossistema Arch Linux para treinamento.
- **Character-Level Tokenizer:** Um tradutor que mapeia caracteres únicos para índices numéricos (IDs) de forma determinística.
- **Pipeline de Dados:** Serialização de texto para tensores binários (`.pt`) otimizados para o PyTorch.
- **Testes Unitários:** Suite de testes com `pytest` garantindo a integridade do processo de encode/decode.

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

Para uma explicação técnica de cada etapa, confira a pasta `/docs`:
- [01: Ingestão de Dados e Tokenização (PT-BR)](docs/pt-br/01-data-ingestion-and-tokenization.md)

---

> "Entender a base é o que diferencia o engenheiro do utilizador de ferramentas."
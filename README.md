# arch-tensor-lab 🚀

> 🇧🇷 **Este README está disponível em [Português](./README.pt-BR.md).**

Welcome to **arch-tensor-lab**. This is my experimentation space focused on building a Large Language Model (LLM) architecture from absolute scratch.
My goal is pedagogical and hands-on: to understand the math, data engineering, and Transformer implementation, starting from the foundation.

---

## 🛑 Project Philosophy

- **From Scratch:** I implement every component (Tokenization, Embeddings, Attention, Backpropagation) to consolidate learning.
- **CPU-First:** Developed in an Arch Linux environment optimized for CPU (Ryzen 5 5600G). Aiming to be a guide for efficiency.
- **"Inca" Coding Style:** Every line of code is written manually. AI assistants are used only to speed up repetitive tasks.

---

## 🛠️ Current Status

### ✅ Phase 01: Data Ingestion and Tokenization
- **Synthetic Dataset:** ~3.5MB of Bash commands and Arch Linux logic.
- **Character-Level Tokenizer:** Deterministic mapping of characters to IDs.
- **Unit Tests:** Full suite using `pytest`.

### ✅ Phase 02: DataLoader and Batch Dynamics
- **Training Matrix:** Implementation of slicing logic ($32 \times 64$).
- **Semantic Sampling:** Using `torch.randint` with safety margins to prevent index overflow.
- **Orchestration:** `run.sh` script and `main.py` for pipeline automation.

---

## 📂 Repository Structure

```text
arch-tensor-lab/
├── data/           # Datasets (raw and processed)
├── docs/           # Detailed documentation in EN/PT-BR
├── src/            # Source code (Tokenizer, Loader, Model)
├── tests/          # Automated tests (pytest)
└── pyproject.toml  # Dependency and environment management
```

---

## 🚀 How to Run

If you are in an Arch Linux environment (or any Linux distro) and want to replicate the lab locally:

1. **Clone the repository:**
```bash
git clone https://github.com/leonardopolicarpo/arch-tensor-lab.git
cd arch-tensor-lab
```

2. **Create the virtual environment and install dependencies:**
*(Note: The lab uses the CPU-only version of PyTorch to save disk and GPU resources)*
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

3. **Run the tests:**
```bash
pytest
```

---

## 📖 Detailed Documentation

Check out the deep technical explanation for each stage:

- **01: Data Ingestion and Tokenization** — [English](docs/en/01-data-ingestion-and-tokenization.md) | [Português](docs/pt-br/01-data-ingestion-and-tokenization.md)
- **02: DataLoader and Batches** — [English](docs/en/02-dataloader.md) | [Português](docs/pt-br/02-dataloader.md)

---

> "Understanding the foundation is what differentiates the engineer from the tool user."
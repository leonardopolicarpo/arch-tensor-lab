# Data Ingestion and Tokenization

This document details the first stage of the **arch-tensor-lab**, which involves creating a synthetic dataset based on the Arch Linux ecosystem and implementing a character-level Tokenizer.

---

## 1. The Raw Material

To train an AI to understand logic and syntax, I chose Bash terminal commands. Unlike human natural language, Bash has a **syntactic rigidity** that facilitates the initial learning process for the neural network.

### Why Bash and Arch Linux?
- **Determinism:** Commands like `pacman -Syu` or `find . -type f` follow clear rules for flags and arguments.
- **Practical Utility:** The model learns to "think" within the environment where it is being developed.
- **Controlled Volume:** I generated approximately **3.5MB** of text (around 5,000 snippets), which is ideal for fast CPU-based testing.
- **Personal Use:** I use the Arch ecosystem on a daily basis.

---

## 2. Tokenizer Architecture

I opted for a **Character-Level Tokenizer** where, instead of whole words, the individual character is the minimum processing unit.



### Technical Implementation:
- **Uniqueness and Determinism:** I used `set()` to extract unique characters and `sorted()` to ensure that each character's index (ID) remains consistent across every execution.
- **Mapping:** I created two lookup dictionaries:
  - `string_to_index`: Translates letters/symbols into numbers for the neural network.
  - `index_to_string`: Translates numbers back into readable text.
- **Efficiency:** The processed text is converted into a **PyTorch Tensor** (`torch.long`) and saved in a `.pt` binary format, ensuring ultra-fast loading during training.

---

## 3. Engineering Challenges and Solutions

I faced some infrastructure obstacles that shaped the project's architecture:

### The Disk Bottleneck
During dependency installation, PyTorch attempted to download gigabytes of CUDA (Nvidia) drivers. Since the lab is **CPU-first**, I resolved this by:
1. `pip cache purge`: Cleaning temporary files to free up space.
2. `pip install torch --index-url https://download.pytorch.org/whl/cpu`: Forcing the installation of a lightweight version, optimized solely for processors.

### Package Structure (src-layout)
I adopted the industrial `src/` pattern to isolate the source code from tests and documentation. This required configuring `pyproject.toml` with `tool.setuptools.packages.find` so that modules would be correctly recognized by the virtual environment.

---

## 4. Quality Assurance

I implemented unit tests using **Pytest** from the start.
- **Isolation:** It uses *fixtures* to create temporary datasets, ensuring that tests do not rely on external files.
- **Integrity:** It validates that the `encode -> decode` process returns the exact original string without any loss of information.
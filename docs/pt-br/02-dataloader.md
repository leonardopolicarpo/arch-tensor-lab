# 02: O DataLoader e a Dinâmica de Batches

> 🌐 **Disponível em outros idiomas:** [English](../en/02-dataloader.md)

Aqui detalho a implementação do **DataLoader**, o componente responsável por fatiar o dataset binário e organizar o fluxo de treinamento em matrizes multidimensionais (Tensores).

---

## 1. O Conceito de Fatiamento

Para que o modelo aprenda, não posso entregar o arquivo de 3.5MB de uma vez. Precisei criar um sistema que extraísse pequenas sequências aleatórias para processamento paralelo.

### Hiperparâmetros Definidos:
- **`batch_size` (32):** Quantas sequências processamos simultaneamente. É o que permite usar o poder vetorial do processador.
- **`block_size` (64):** O tamanho da "janela de contexto". A rede olha para 64 caracteres para tentar prever o 65º.

---

## 2. A Lógica de Sorteio e Stacking

O processo de transformar uma fita linear de dados em uma matriz de treinamento envolve três passos fundamentais:

### A. Sorteio de Índices (`torch.randint`)
Utilizei a função `randint` para gerar pontos de partida aleatórios no tensor. Para evitar que o fatiamento tentasse ler dados além do fim do arquivo (causando erro de índice), apliquei uma **margem de segurança**:
`limite_maximo = len(data) - block_size - 1`.

### B. O Deslocamento X e Y
A rede neural aprende por predição de próximo caractere. Por isso, para cada entrada **X**, gerei um alvo **Y** que é exatamente a mesma sequência, mas deslocada um índice para a direita.

### C. O "Grampeador" (`torch.stack`)
Cada sequência fatiada é inicialmente um tensor 1D. O `torch.stack` atua como um grampeador, empilhando essas 32 sequências verticalmente para formar uma matriz **32x64**. Isso otimiza o cálculo matemático, permitindo que a CPU processe o bloco inteiro de uma vez.

---

## 3. Visualização Matemática

Para um exemplo simplificado de `batch_size=2` e `block_size=4`, a estrutura gerada pelo meu DataLoader segue este modelo:

**Input (X):**
$$
\begin{bmatrix} 
10 & 22 & 15 & 40 \\
55 & 12 & 33 & 08 
\end{bmatrix}
$$

**Target (Y):**
$$
\begin{bmatrix} 
22 & 15 & 40 & 11 \\
12 & 33 & 08 & 99 
\end{bmatrix}
$$

Dessa forma, quando a rede encontra o token `10`, ela é treinada para saber que o próximo deve ser `22`. Quando vê a sequência `10, 22, 15`, o alvo é `40`.

---

## 4. Eficiência de Memória

Ao carregar o arquivo `.pt` com `weights_only=True`, garanti que apenas os dados puros fossem alocados na RAM/CPU. A utilização de tensores `int64` (Long) é vital aqui, pois as funções de perda (*loss functions*) e camadas de *embedding* do PyTorch exigem esse formato para indexação.

---

## 5. Insights

Durante a construção do DataLoader, ficou claro que os tokens (`int64`) que estou manipulando são, na verdade, endereços de memória para o que virá a seguir:

1. **A Tabela de Embeddings:** Cada ID servirá para "pescar" um vetor de números decimais (ex: $v = [0.12, -0.5, \dots]$) que representa o caractere em um espaço vetorial.
2. **Conexão com VectorDBs:** Essa lógica é a base dos bancos de dados vetoriais. Enquanto meu DataLoader prepara índices para *previsão*, um VectorDB utiliza esses mesmos tipos de vetores para *busca semântica* (encontrar textos com significados parecidos).
3. **Escalabilidade:** Entender isso no nível de caractere me dá a base para, no futuro, implementar tokenização por palavras ou sub-palavras (BPE), onde o "significado" extraído será muito mais denso.

---

> "O DataLoader é a ponte entre o dado estático e o aprendizado dinâmico."
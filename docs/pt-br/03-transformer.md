# 03: O Transformer

> 🌐 **Disponível em outros idiomas:** [English](../en/03-transformer.md)

## 1. Visão Geral:

A classe **LanguageModel** é o coração matemático do Small Language Model(SLM). Ela herda de **torch.nn.Module**, que é a classe base do PyTorch para a construção de qualquer rede neural.

* _\_\_init\_\__: declaro as matrizes de pesos, sendo a tabela de *Embeddings* e a camada *Linear*, e aloco espaço na memória, preparando a infra, mas nao processando os dados.
Ao chamar o inicializador da classe mãe (**nn.Module**), o modelo é registrado, permitindo que o framework rastreie os parâmetros internamente, construa o grafo computacional para o cálculo de derivadas (Autograd) no momento do treinamento e permita a exportação de pesos finais(**\.pt**) para uso futuro.

* _forward_: rota obrigatória por onde os tensores de dados fluem, camada por camada, até se transformarem em uma predição final

## 2. O Construtor

É criada a 'física' do modelo, alocando na memória as duas estruturas matemáticas que farão a tradução entre o que a rede "lê", "pensa" e "fala".

### 2.1. **nn.Embedding** (A tabela de significados)

* O problema: Redes neurais não entendem letras ou IDs numéricos. Se passar ID 4 e ID 2 para a rede, a matemática vai assumir erroneamente que um vale o dobro do outro. Eles são apenas categorias, não grandezas. Preciso transformar categorias em coordenadas espaciais.

* A solução: O **nn.Embedding** cria uma tabela de busca de alta velocidade (um Lookup Table). Alocada na memória RAM, onde ela tem **V** linhas (onde **V** é o *vocabulary_size*, ex: 62) e **D** colunas (a *embedding_dimension*, ex: 128)

* Quando o PyTorch recebe o token de ID 5, ele não faz multiplicações matriciais, mas sim vai diretamente na linha 5 dessa tabela e pega um vetor inteiro contendo 128 números decimais. Esses 128 números são as coordenadas daquele token no espaço de raciocínio da IA. É assim que a rede abstrai o 'significado' de um caractere.

### 2.2. **nn.Linear** (A cabeça da predição / LM Head)

* O problema: O modelo foi configurado então para 'pensar' usando 128 dimensões (**D**), mas preciso que ele me dê uma resposta múltipla escolha baseada nas 62 opções de vocabulário (**V**)

* A solução: A camada **nn.Linear** (*language_modeling_head*) atua como um tradutor / espécie de funil, onde aplica uma transformação linear (y = xA^T + b) para projetar o espaço vetorial interno de volta para o espaço do vocabulário.

* Ela recebe o vetor de 128 dimensões gerado pelo Embedding e o transforma até ter 62 números, sendo eles chamados de *logits*, representam a 'força do chute' (a nota) da IA para cada um dos caracteres possíveis. O que tiver maior nota é a predição de qual a próxima letra da sequência.
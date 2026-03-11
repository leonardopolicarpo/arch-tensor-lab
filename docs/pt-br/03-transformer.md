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

## 3. O Fluxo de Dados (**forward**) e o Cálculo de Erro

O método **forward** recebe os IDs brutos e os transforma em predições, calculando o quão longe a IA está da resposta correta.

### 3.1. A Viagem do Tensor (o caminho de ida)

1. Entrada: o método recebe *input_indices* (contexto **X**), sendo o formato dessa matriz bidimensional: **[batch_size, sequence_length]** (ex: **[32, 64]**)

2. Ganho de profundidade: Ao passar **X** pela *token_embedding_table*, cada número simples vira um vetor de 128 dimensões. O tensor se transforma num cubo 3D com formato **[batch_size, sequence_length, embedding_dimension]**

3. 'Afunilamento' para resposta: Esse 'cubo' passa pela camada *Linear* (**language_modeling_head**), onde ela comprime essa dimensão de volta para o tamanho do vocabulário (62). O formato final das notas de predição (**logits**) é **[batch_size, sequence_length, vocabulary_size]**

### 3.2. A Matemática da Perda (loss calculation)

Se passar a matriz **targets** (o gabarito **Y**) para o **forward**, a rede vai calcular a sua 'nota de prova', que é chamada de Perda.

- O problema: A função de avaliação do PyTorch exige uma lista bidimensional contínua, e nossas matrizes foram geradas 3D **[B, T, C]**

- A solução: uso o método *view()* para reestruturar o tensor sem alterar os dados na memória, pegando as 32 sequências de comandos (cada uma com 64 caracteres de comprimento) e empilhamos tudo numa única fila contínua
  - *logits* vira **[batch_size * sequence_length, channels]** -> **[2048, 62]**
  - *targets* vira **[batch_size * sequence_length]** -> **[2048]**

- *cross_entropy*: com essas matrizes o PyTorch compara as 62 probabilidades geradas pela rede com a resposta correta e nos devolve um número escalar (ex: 4.24), onde quanto menor esse número, mais inteligente a rede está ficando, sendo que uma *Loss* alta significa que a rede está chutando aleatoriamente
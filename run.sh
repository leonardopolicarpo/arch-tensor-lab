#!/bin/bash

VERMELHO=$(tput setaf 1)
VERDE=$(tput setaf 2)
AMARELO=$(tput setaf 3)
AZUL=$(tput setaf 4)
NEGRITO=$(tput bold)
RESET=$(tput sgr0)

clear

echo "${AZUL}---------------------------------------------------${RESET}"
echo "  ${NEGRITO}${VERMELHO}Bem-vindo ao Arch-Tensor-Lab${RESET}"
echo "${AZUL}---------------------------------------------------${RESET}"
echo ""

echo "${AMARELO}>> Configuração do Projeto${RESET}"

read -p "${NEGRITO}Nome do Dataset${RESET} (press enter for default)${NEGRITO}:${RESET} " DATASET

if [ -z "$DATASET" ]; then
  echo "bash_commands selecionado"
  DATASET="bash_commands"
fi

read -p "${NEGRITO}Nome do Binário de Destino${RESET} (press enter for default)${NEGRITO}:${RESET} " BINARIO

if [ -z "$BINARIO" ]; then
  echo "bash_data selecionado"
  BINARIO="bash_data"
fi

read -p "${NEGRITO}Nome do Modelo${RESET} (press enter to match dataset)${NEGRITO}:${RESET} " MODELO

if [ -z "$MODELO" ]; then
  MODELO="$DATASET"
  echo "Modelo nomeado como: $MODELO"
fi

read -p "${NEGRITO}Deseja treinar o modelo agora?${RESET} (s/N)${NEGRITO}:${RESET} " TREINAR

TRAIN_FLAG=""
if [[ "$TREINAR" == "s" || "$TREINAR" == "S" ]]; then
  TRAIN_FLAG="--train"
fi

echo ""
echo "${VERDE}✔ Configurações salvas com sucesso!${RESET}"
echo "Processando: ${AZUL}$DATASET${RESET} -> ${AZUL}$BINARIO${RESET}"
echo "Dataset: ${AZUL}$DATASET${RESET} | Binário: ${AZUL}$BINARIO${RESET} | Pesos: ${AZUL}${MODELO}_weights.pt${RESET}"
echo "Modo de Treino: ${AZUL}$(if [ -n "$TRAIN_FLAG" ]; then echo "ATIVADO"; else echo "DESATIVADO (Apenas Geração)"; fi)${RESET}"
echo "---------------------------------------------------"

python main.py -f "${DATASET}.txt" -t "${BINARIO}.pt" -m "$MODELO" $TRAIN_FLAG
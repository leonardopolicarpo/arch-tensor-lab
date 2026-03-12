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

read -p "${NEGRITO}Nome do Dataset${RESET} [chat_pro]: " DATASET
DATASET=${DATASET:-"chat_pro"}

read -p "${NEGRITO}Nome do Binário (.pt)${RESET} [agent_v3]: " BINARIO
BINARIO=${BINARIO:-"agent_v3"}

read -p "${NEGRITO}Nome do Modelo/Pesos${RESET} [agent_v3]: " MODELO
MODELO=${MODELO:-"agent_v3"}

read -p "${NEGRITO}Deseja treinar o modelo agora?${RESET} (s/N): " TREINAR

TRAIN_FLAG=""
STEPS_FLAG=""

if [[ "$TREINAR" == "s" || "$TREINAR" == "S" ]]; then
  TRAIN_FLAG="--train"
  
  read -p "${NEGRITO}Quantidade de Passos${RESET} [2000]: " PASSOS
  PASSOS=${PASSOS:-2000}
  STEPS_FLAG="--steps $PASSOS"
fi

echo ""
echo "${VERDE}✔ Configurações confirmadas!${RESET}"
echo "Dataset: ${AZUL}${DATASET}.txt${RESET}"
echo "Binário: ${AZUL}${BINARIO}.pt${RESET}"
echo "Pesos  : ${AZUL}${MODELO}_weights.pt${RESET}"

if [ -n "$TRAIN_FLAG" ]; then
  echo "Modo   : ${VERMELHO}TREINAMENTO ATIVADO ($PASSOS passos)${RESET}"
else
  echo "Modo   : ${AZUL}APENAS INFERÊNCIA (Geração)${RESET}"
fi
echo "---------------------------------------------------"

python main.py -f "${DATASET}.txt" -t "${BINARIO}.pt" -m "$MODELO" $TRAIN_FLAG $STEPS_FLAG
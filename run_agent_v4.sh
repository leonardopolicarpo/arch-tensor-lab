#!/bin/bash

VERMELHO=$(tput setaf 1)
VERDE=$(tput setaf 2)
AMARELO=$(tput setaf 3)
AZUL=$(tput setaf 4)
NEGRITO=$(tput bold)
RESET=$(tput sgr0)

clear

echo "${AZUL}---------------------------------------------------${RESET}"
echo "  ${NEGRITO}${VERMELHO}Arch-Tensor-Lab: Motor Agent V4${RESET}"
echo "${AZUL}---------------------------------------------------${RESET}"
echo ""

echo "${AMARELO}>> Configuração do Ciclo de Treino${RESET}"

read -p "${NEGRITO}Deseja iniciar/continuar o treino?${RESET} (s/N): " TREINAR

TRAIN_FLAG=""
STEPS_FLAG=""

if [[ "$TREINAR" == "s" || "$TREINAR" == "S" ]]; then
  TRAIN_FLAG="--train"
  
  read -p "${NEGRITO}Quantidade de passos para este ciclo${RESET} [2000]: " PASSOS
  PASSOS=${PASSOS:-2000}
  STEPS_FLAG="--steps $PASSOS"
fi

echo ""
echo "${VERDE}✔ Configurações carregadas!${RESET}"
echo "Dataset : ${AZUL}chat_pro.txt${RESET}"
echo "Modelo  : ${AZUL}agent_v4_weights.pt${RESET}"

if [ -n "$TRAIN_FLAG" ]; then
  echo "Ação    : ${VERMELHO}TREINAR (${PASSOS} passos)${RESET}"
  echo "Status  : ${AMARELO}Checkpoints ativos a cada 500 passos${RESET}"
else
  echo "Ação    : ${AZUL}APENAS INFERÊNCIA (Teste de Geração)${RESET}"
fi

echo "---------------------------------------------------"

python main.py -f "chat_pro.txt" -t "agent_v4.pt" -m "agent_v4" $TRAIN_FLAG $STEPS_FLAG
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

read -p "${NEGRITO}Deseja treinar o modelo agora?${RESET} (s/N)${NEGRITO}:${RESET} " TREINAR

TRAIN_FLAG=""
if [[ "$TREINAR" == "s" || "$TREINAR" == "S" ]]; then
  TRAIN_FLAG="--train"
fi

echo ""
echo "${VERDE}✔ Configurações salvas com sucesso!${RESET}"
echo "Modo de Treino: ${AZUL}$(if [ -n "$TRAIN_FLAG" ]; then echo "ATIVADO"; else echo "DESATIVADO (Apenas Geração)"; fi)${RESET}"
echo "---------------------------------------------------"

python main.py -f "chat_pro.txt" -t "agent_v2.pt" -m "agent_v2" $TRAIN_FLAG
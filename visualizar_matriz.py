# MEMBROS DO GRUPO:
# André Portela Lino - 15634885
# Davi Lima de Oliveira - 15648741
# Eduardo Almeida Cavalcanti de Melo - 15526004
# Eric Isin Wang Chou - 15574579
# Júlio Arroio Silva - 15466241
# Karina Yang Chen - 15466658
# TURMA 94

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

caminho_matriz = "saidas/matriz_confusao.txt"
caminho_salvamento = "saidas/matriz_confusao_visual.png"

if not os.path.exists(caminho_matriz):
    print(f"[ERRO] O arquivo '{caminho_matriz}' nao foi encontrado!")
    print("Por favor, execute o seu 'multilayer_perceptron.py' primeiro para gerar os dados.")
    exit()

matriz = np.loadtxt(caminho_matriz, dtype=int)

# cria uma lista automatica com as letras do alfabeto para os eixos
alfabeto = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
plt.figure(figsize=(14, 11))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", 
            xticklabels=alfabeto, yticklabels=alfabeto,
            linewidths=0.5, linecolor="gray")

# titulos e rotulos
plt.title("Matriz de Confusão - Reconhecimento de Caracteres (MLP)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Letra Predita pela Rede Neural", fontsize=13, labelpad=10)
plt.ylabel("Letra Real (Alvo Correto)", fontsize=13, labelpad=10)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# salva automaticamente
plt.savefig(caminho_salvamento, dpi=300)
print(f"[SUCESSO] O gráfico da matriz de confusão foi salvo em: '{caminho_salvamento}'")

# exibe a janela na tela 
plt.show()
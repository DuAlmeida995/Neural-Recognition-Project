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

caminho_erros = "saidas/historico_erros.txt"
caminho_salvamento = "saidas/historico_erros_visual.png"

if not os.path.exists(caminho_erros):
    print(f"[ERRO] O arquivo '{caminho_erros}' nao foi encontrado!")
    print("Por favor, execute o seu 'multilayer_perceptron.py' primeiro para gerar os dados.")
    exit()

arranjo = np.loadtxt(caminho_erros, dtype=float)

plt.figure(figsize=(10, 6))
plt.plot(arranjo, label="Erro Quadrático Médio", color="blue", marker="o", markersize=4, linewidth=2)
plt.title("Histórico de Erros - Reconhecimento de Caracteres (MLP)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Iteração", fontsize=13, labelpad=10)
plt.ylabel("Erro Quadrático Médio", fontsize=13, labelpad=10)
plt.legend()
plt.tight_layout()
plt.savefig(caminho_salvamento, dpi=300)
print(f"[SUCESSO] O gráfico do histórico de erros foi salvo em: '{caminho_salvamento}'")
plt.show()
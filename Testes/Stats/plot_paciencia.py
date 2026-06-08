import numpy as np
import matplotlib.pyplot as plt

dataAlpha = np.load("Testes/Stats/statsAlpha.npy") * 100
dataCamada = np.load("Testes/Stats/statsCamadas.npy") * 100
dataParada = np.load("Testes/Stats/statsPaciencia.npy") * 100

plt.figure(figsize=(10, 6))
plt.plot(dataParada, label='Variação do Número de Paciência', marker='o')
plt.xlabel('Número de Paciência')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia em função do Número de Paciência')
xParada = range(1, len(dataParada)+1)
plt.xticks(range(len(dataParada)), np.array(xParada)+9, rotation=45)
plt.grid()
plt.legend()
plt.savefig("Testes/Stats/plot_paciencia.png")
plt.show()
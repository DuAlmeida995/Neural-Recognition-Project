import numpy as np
import matplotlib.pyplot as plt

dataAlpha = np.load("Testes/Stats/statsAlpha.npy") * 100
dataCamada = np.load("Testes/Stats/statsCamadas.npy") * 100
dataParada = np.load("Testes/Stats/statsParadas.npy") * 100

plt.figure(figsize=(10, 6))
plt.plot(dataAlpha, label='Variação do Alpha', marker='o')
plt.xlabel('Taxa de aprendizado (Alpha)')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia em função da Taxa de Aprendizado (Alpha)')
xAlfa = range(1, len(dataAlpha)+1)
plt.xticks(range(len(dataAlpha)), 0.01*np.array(xAlfa), rotation=45)
plt.grid()
plt.legend()
plt.savefig("Testes/Stats/plot_alpha.png")
plt.show()
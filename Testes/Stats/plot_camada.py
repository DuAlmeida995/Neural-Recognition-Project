import numpy as np
import matplotlib.pyplot as plt

dataAlpha = np.load("Testes/Stats/statsAlpha.npy") * 100
dataCamada = np.load("Testes/Stats/statsCamadas.npy") * 100
dataParada = np.load("Testes/Stats/statsParadas.npy") * 100

plt.figure(figsize=(10, 6))
plt.plot(dataCamada, label='Variação do Número de Camadas', marker='o')
plt.xlabel('Número de Camadas')
plt.ylabel('Acurácia (%)')
plt.title('Acurácia em função do Número de Camadas')
xCamada = range(1, len(dataCamada)+1)
plt.xticks(range(len(dataCamada)), np.array(xCamada)*10, rotation=45)
plt.grid()
plt.legend()
plt.savefig("Testes/Stats/plot_camada.png")
plt.show()
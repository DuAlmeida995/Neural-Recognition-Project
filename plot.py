# MEMBROS DO GRUPO:
# André Portela Lino - 15634885
# Davi Lima de Oliveira - 15648741
# Eduardo Almeida Cavalcanti de Melo - 15526004
# Eric Isin Wang Chou - 15574579
# Júlio Arroio Silva - 15466241
# Karina Yang Chen - 15466658
# TURMA 94

# plota resultados dos experimentos — α, camada escondida e épocas (Apenas Sigmóide)

import numpy as np
import matplotlib.pyplot as plt

# Remove experimentos que não convergiram (acuracia=0) pra não distorcer os gráficos
# épocas: escondida=70, pesos aleatórios
dataSigRand170 = np.delete(np.load("Testes/estatisticas/mlp_sig_rand_1_70.npy"), np.load("Testes/estatisticas/mlp_sig_rand_1_70.npy")[:, 1] == 0, axis=0)
dataSigRand270 = np.delete(np.load("Testes/estatisticas/mlp_sig_rand_2_70.npy"), np.load("Testes/estatisticas/mlp_sig_rand_2_70.npy")[:, 1] == 0, axis=0)

# Experimentos por hiperparâmetros (busca em grade: α × camada escondida)
# Coluna [7]=acurácia no teste — remove experimentos com acurácia ≤ 0.01
dataAleatoriosSigmoide = np.delete(np.load("Testes/estatisticas/pesosAleatoriosSigmoide.npy"), np.load("Testes/estatisticas/pesosAleatoriosSigmoide.npy")[:, 7] <= 0.01, axis=0)
dataZeradosSigmoide = np.delete(np.load("Testes/estatisticas/pesosZeradosSigmoide.npy"), np.load("Testes/estatisticas/pesosZeradosSigmoide.npy")[:, 7] <= 0.01, axis=0)

# cols: [0]=alpha, [1]=escondida, [2]=acertos, [3]=erros, [4]=acur. total, [5]=acertos teste, [6]=erros teste, [7]=acur. teste
alfaAleatoriosSigmoide = dataAleatoriosSigmoide[:, 0]
camadaEscondidaAleatoriosSigmoide = dataAleatoriosSigmoide[:, 1]
acuraciaTreinamentoAleatoriosSigmoide = dataAleatoriosSigmoide[:, 7]

alfaZeradosSigmoide = dataZeradosSigmoide[:, 0]
camadaEscondidaZeradosSigmoide = dataZeradosSigmoide[:, 1]
acuraciaTreinamentoZeradosSigmoide = dataZeradosSigmoide[:, 7]

# Menu simplificado: usuário escolhe a dimensão temporal ou estrutural
epocaOuAlfaCamada = input("Digite o que deseja plotar (alfa/camada escondida[0] ou epoca[1]): ")

if epocaOuAlfaCamada == '0':
    # Modo de análise por hiperparâmetros (α e/ou tamanho da camada escondida)
    formatoPesos = input("Digite o formato dos pesos (zerados[0] ou aleatórios[1]): ")
    operacaoGrafica = input("Digite a operação gráfica (alfa[0], camada escondida[1] ou ambos[2]): ")

    if operacaoGrafica == '2':
        # 2D + 3D: superfície de desempenho α × camada escondida
        fig = plt.figure()

        if formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            x = alfaZeradosSigmoide
            y = camadaEscondidaZeradosSigmoide
        else:
            z = acuraciaTreinamentoAleatoriosSigmoide
            x = alfaAleatoriosSigmoide
            y = camadaEscondidaAleatoriosSigmoide

        ax = fig.add_subplot(2,2,1)
        ax.scatter(x, z, c=z, cmap='viridis')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Acurácia de Treinamento')
        ax.set_title('Acurácia de Treinamento vs Alpha')

        ax = fig.add_subplot(2,2,2)
        ax.scatter(y, z, c=z, cmap='viridis')
        ax.set_xlabel('Camada Escondida')
        ax.set_ylabel('Acurácia de Treinamento')
        ax.set_title('Acurácia de Treinamento vs Camada Escondida')

        ax = fig.add_subplot(2,2,3, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Camada Escondida')
        ax.set_zlabel('Acurácia de Treinamento')
        ax.set_title('Acurácia de Treinamento vs Alpha e Camada Escondida')
        
        array = np.array([x, y, z])

    elif operacaoGrafica == '0':
        fig = plt.figure()

        if formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            x = alfaZeradosSigmoide
        else:
            z = acuraciaTreinamentoAleatoriosSigmoide
            x = alfaAleatoriosSigmoide

        plt.plot(x, z, 'o', color='blue')
        plt.xlabel('Alpha')
        plt.ylabel('Acurácia de Treinamento')
        plt.title('Acurácia de Treinamento vs Alpha')

        array = np.array([x, z])

    elif operacaoGrafica == '1':
        fig = plt.figure()

        if formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            y = camadaEscondidaZeradosSigmoide
        else:
            z = acuraciaTreinamentoAleatoriosSigmoide
            y = camadaEscondidaAleatoriosSigmoide

        plt.plot(y, z, 'o', color='blue')
        plt.xlabel('Camada Escondida')
        plt.ylabel('Acurácia de Treinamento')
        plt.title('Acurácia de Treinamento vs Camada Escondida')

        array = np.array([y, z])
else:
    # Modo épocas: vê a partir de quantas épocas a acurácia se estabiliza para a Sigmóide
    fig = plt.figure()

    # Dois subplots = duas seeds — checa estabilidade dos resultados
    x1 = dataSigRand170[:, 0]   # número de épocas
    y1 = dataSigRand170[:, 1]   # acurácia correspondente

    ax = fig.add_subplot(2,2,1)
    ax.scatter(x1, y1, c=y1, cmap='viridis')
    ax.set_xlabel('Epocas')
    ax.set_ylabel('Acurácia de Treinamento')
    ax.set_title('Acurácia de Treinamento vs Epocas')

    x2 = dataSigRand270[:, 0]
    y2 = dataSigRand270[:, 1]

    ax = fig.add_subplot(2,2,2)
    ax.scatter(x2, y2, c=y2, cmap='viridis')
    ax.set_xlabel('Epocas')
    ax.set_ylabel('Acurácia de Treinamento')
    ax.set_title('Acurácia de Treinamento vs Epocas')

    array1 = np.array([x1, y1])
    array1 = array1.transpose()
    print(array1)

    array = np.array([x2, y2])

array = array.transpose()
print(array)

plt.show()

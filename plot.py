# =============================================================================
# Visualização dos Resultados dos Experimentos com a MLP
# =============================================================================
# Este script carrega os dados de estatísticas gravados pelos scripts de teste
# e gera gráficos interativos para analisar o impacto dos hiperparâmetros
# sobre a acuíral da rede no conjunto de teste.
#
# Experimentos disponíveis para visualização:
#   - Variação de α (taxa de aprendizado) e tamanho da camada escondida
#     com pesos inicializados aleatoriamente ou zerados (sigmoide ou tanh)
#   - Variação do número de épocas com camada escondida fixada em 70 neurônios
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Carregamento e filtragem dos dados de estatísticas
# -----------------------------------------------------------------------------
# Cada arquivo .npy contém os resultados de múltiplas rodadas de treinamento
# variando hiperparâmetros. Usamos np.delete para remover linhas onde a
# acuíacia é zero ou muito baixa (experimentos que não convergiram),
# para que esses outliers não distorcem as escalas dos gráficos.

# Experimentos por épocas: camada escondida=70 neurônios, pesos aleatórios
# Colunas [0]=número de épocas, [1]=acuíacia — remove linhas com acuíacia=0
dataSigRand170 = np.delete(np.load("Testes/estatisticas/mlp_sig_rand_1_70.npy"),np.load("Testes/estatisticas/mlp_sig_rand_1_70.npy")[:, 1] == 0, axis=0)
dataSigRand270 = np.delete(np.load("Testes/estatisticas/mlp_sig_rand_2_70.npy"),np.load("Testes/estatisticas/mlp_sig_rand_2_70.npy")[:, 1] == 0, axis=0)
dataTanhRand170 = np.delete(np.load("Testes/estatisticas/mlp_tanh_random_1_70.npy"),np.load("Testes/estatisticas/mlp_tanh_random_1_70.npy")[:, 1] == 0, axis=0)
dataTanhRand270 = np.delete(np.load("Testes/estatisticas/mlp_tanh_random_2_70.npy"),np.load("Testes/estatisticas/mlp_tanh_random_2_70.npy")[:, 1] == 0, axis=0)

# Experimentos por hiperparâmetros (busca em grade: α × camada escondida)
# Coluna [7]=acuíacia no teste — remove experimentos com acuíacia ≤ 0.01
dataAleatoriosTanh = np.delete(np.load("Testes/estatisticas/pesosAleatoriosTanh.npy"),np.load("Testes/estatisticas/pesosAleatoriosTanh.npy")[:, 7] <= 0.01, axis=0)
dataAleatoriosSigmoide = np.delete(np.load("Testes/estatisticas/pesosAleatoriosSigmoide.npy"),np.load("Testes/estatisticas/pesosAleatoriosSigmoide.npy")[:, 7] <= 0.01, axis=0)
dataZeradosTanh = np.delete(np.load("Testes/estatisticas/pesosZeradosTanh.npy"),np.load("Testes/estatisticas/pesosZeradosTanh.npy")[:, 7] <= 0.01, axis=0)
dataZeradosSigmoide = np.delete(np.load("Testes/estatisticas/pesosZeradosSigmoide.npy"),np.load("Testes/estatisticas/pesosZeradosSigmoide.npy")[:, 7] <= 0.01, axis=0)


# -----------------------------------------------------------------------------
# Extração das colunas de interesse de cada conjunto de dados
# -----------------------------------------------------------------------------
# Layout das colunas nos arquivos de busca em grade:
#   [0]=Alpha, [1]=tam. camada escondida, [2]=acertos totais, [3]=erros totais,
#   [4]=acuíacia total, [5]=acertos teste, [6]=erros teste, [7]=acuíacia teste
alfaAleatoriosTanh = dataAleatoriosTanh[:, 0]
camadaEscondidaAleatoriosTanh = dataAleatoriosTanh[:, 1]
acuraciaTreinamentoAleatoriosTanh = dataAleatoriosTanh[:, 7]

alfaAleatoriosSigmoide = dataAleatoriosSigmoide[:, 0]
camadaEscondidaAleatoriosSigmoide = dataAleatoriosSigmoide[:, 1]
acuraciaTreinamentoAleatoriosSigmoide = dataAleatoriosSigmoide[:, 7]

alfaZeradosTanh = dataZeradosTanh[:, 0]
camadaEscondidaZeradosTanh = dataZeradosTanh[:, 1]
acuraciaTreinamentoZeradosTanh = dataZeradosTanh[:, 7]

alfaZeradosSigmoide = dataZeradosSigmoide[:, 0]
camadaEscondidaZeradosSigmoide = dataZeradosSigmoide[:, 1]
acuraciaTreinamentoZeradosSigmoide = dataZeradosSigmoide[:, 7]

# -----------------------------------------------------------------------------
# Menu interativo: o usuário escolhe o que deseja visualizar
# -----------------------------------------------------------------------------
epocaOuAlfaCamada = input("Digite o que deseja plotar (alfa/camada escondida[0] ou epoca[1]): ")

if epocaOuAlfaCamada == '0':
    # Modo de análise por hiperparâmetros (α e/ou tamanho da camada escondida).
    # O usuário escolhe a função de ativação, o tipo de inicialização dos pesos
    # e qual dimensão quer analisar graficamente.
    funcaoAtivacao = input("Digite a função de ativação (tanh[0] ou sigmoide[1]): ")
    formatoPesos = input("Digite o formato dos pesos (zerados[0] ou aleatórios[1]): ")
    operacaoGrafica = input("Digite a operação gráfica (alfa[0], camada escondida[1] ou ambos[2]): ")

    if operacaoGrafica == '2':
        # Visualização completa: dois gráficos 2D (Alpha vs Acuíacia e
        # Camada Escondida vs Acuíacia) e um gráfico 3D que combina as duas
        # variáveis, formando uma superfície de desempenho da rede.
        # O mapa de cores (viridis) facilita identificar visualmente as
        # combinações de hiperparâmetros com maior acuíacia.
        fig = plt.figure()

        if funcaoAtivacao == '0' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosTanh
            x = alfaZeradosTanh
            y = camadaEscondidaZeradosTanh
        elif funcaoAtivacao == '0' and formatoPesos == '1':
            z = acuraciaTreinamentoAleatoriosTanh
            x = alfaAleatoriosTanh
            y = camadaEscondidaAleatoriosTanh
        elif funcaoAtivacao == '1' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            x = alfaZeradosSigmoide
            y = camadaEscondidaZeradosSigmoide
        elif funcaoAtivacao == '1' and formatoPesos == '1':
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

        if funcaoAtivacao == '0' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosTanh
            x = alfaZeradosTanh
        elif funcaoAtivacao == '0' and formatoPesos == '1':
            z = acuraciaTreinamentoAleatoriosTanh
            x = alfaAleatoriosTanh
        elif funcaoAtivacao == '1' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            x = alfaZeradosSigmoide
        elif funcaoAtivacao == '1' and formatoPesos == '1':
            z = acuraciaTreinamentoAleatoriosSigmoide
            x = alfaAleatoriosSigmoide

        plt.plot(x, z, 'o', color='blue')
        plt.xlabel('Alpha')
        plt.ylabel('Acurácia de Treinamento')
        plt.title('Acurácia de Treinamento vs Alpha')

        array = np.array([x, z])


    elif operacaoGrafica == '1':
        fig = plt.figure()

        if funcaoAtivacao == '0' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosTanh
            y = camadaEscondidaZeradosTanh
        elif funcaoAtivacao == '0' and formatoPesos == '1':
            z = acuraciaTreinamentoAleatoriosTanh
            y = camadaEscondidaAleatoriosTanh
        elif funcaoAtivacao == '1' and formatoPesos == '0':
            z = acuraciaTreinamentoZeradosSigmoide
            y = camadaEscondidaZeradosSigmoide
        elif funcaoAtivacao == '1' and formatoPesos == '1':
            z = acuraciaTreinamentoAleatoriosSigmoide
            y = camadaEscondidaAleatoriosSigmoide

        plt.plot(y, z, 'o', color='blue')
        plt.xlabel('Camada Escondida')
        plt.ylabel('Acurácia de Treinamento')
        plt.title('Acurácia de Treinamento vs Camada Escondida')

        array = np.array([y, z])
else:
    # Modo de análise por número de épocas: mostra como a acuíacia evolui
    # conforme aumentamos o limite máximo de épocas de treinamento,
    # com a camada escondida fixada em 70 neurônios e α = 0.1.
    # Isso ajuda a encontrar o ponto ótimo de treinamento: poucas épocas
    # = underfitting; muitas épocas = risco de overfitting.
    funcaoAtivacao = input("Digite a função de ativação (tanh[0] ou sigmoide[1]): ")
    if funcaoAtivacao == '0':
        fig = plt.figure()

        # Cada subplot corresponde a uma execução com seeds diferentes,
        # permitindo verificar a estabilidade dos resultados (variação por acaso)
        x1 = dataTanhRand170[:, 0]   # número de épocas
        y1 = dataTanhRand170[:, 1]   # acuíacia correspondente

        ax = fig.add_subplot(2,2,1)
        ax.scatter(x1, y1, c=y1, cmap='viridis')
        ax.set_xlabel('Epocas')
        ax.set_ylabel('Acurácia de Treinamento')
        ax.set_title('Acurácia de Treinamento vs Epocas')

        x2 = dataTanhRand270[:, 0]
        y2 = dataTanhRand270[:, 1]

        ax = fig.add_subplot(2,2,2)
        ax.scatter(x2, y2, c=y2, cmap='viridis')
        ax.set_xlabel('Epocas')
        ax.set_ylabel('Acurácia de Treinamento')
        ax.set_title('Acurácia de Treinamento vs Epocas')

        array1 = np.array([x1, y1])
        array1 = array1.transpose()
        print(array1)

        array = np.array([x2, y2])
    else:
        fig = plt.figure()

        x1 = dataSigRand170[:, 0]
        y1 = dataSigRand170[:, 1]

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
print(array)  # imprime os dados plotados no formato (n_experimentos, 2)

# Exibe todos os gráficos gerados na janela do matplotlib
plt.show()
# MEMBROS DO GRUPO:
# André Portela Lino - 15634885
# Davi Lima de Oliveira - 15648741
# Eduardo Almeida Cavalcanti de Melo - 15526004
# Eric Isin Wang Chou - 15574579
# Júlio Arroio Silva - 15466241
# Karina Yang Chen - 15466658
# TURMA 94

import numpy as np # Biblioteca utilizada para entrada e saida dos conjuntos de dados, além de manipulação de arrays e operações matemáticas
import os # Biblioteca utilizada para manipulação de arquivos e diretórios (será utilizada apenas para criar um diretório para escrita de parâmetros)

# Função de ativação usada no feedforward e backpropagation
def sigmoide(x):
    return 1 / (1 + np.exp(-x)) # Retorna a saida de x após a aplicação da função sigmoide

# Derivada da função sigmoide, usada no backpropagation para calcular os deltas
def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x)) # Retorna a derivada da função sigmoide, que é a função sigmoide multiplicada por (1 - função sigmoide)

# Função que retorna a posição do maior elemento de um vetor
def argmax(x):
    for i in range(len(x)): # Itera sobre o vetor x e retorna o índice do maior elemento
        if x[i] == max(x):
            return i
    return -1 # Retorna -1 caso o vetor seja vazio ou não contenha elementos, apenas por segurança

# Classe MLP (Multi-Layer Perceptron) que implementa a rede neural para reconhecimento de caracteres
class MLP:
    # Inicializa a rede neural com os parâmetros especificados
    def __init__(self, n_entrada, n_escondida, n_saida, taxa_aprendizado):
        self.n_entrada = n_entrada # Variável para armazenar o número de neurônios na camada de entrada
        self.n_escondida = n_escondida # Variável para armazenar o número de neurônios na camada escondida
        self.n_saida = n_saida # Variável para armazenar o número de neurônios na camada de saída
        self.alpha = taxa_aprendizado # Variável para armazenar a taxa de aprendizado

        # Inicializa os pesos da camada escondida (V) e da camada de saída (W)
        # Os intervalos de inicialização são baseados no número de neurônios para evitar a saturação das funções de ativação
        self.v = np.random.uniform(-1/self.n_entrada, 1/self.n_entrada, (self.n_escondida, self.n_entrada + 1))
        self.w = np.random.uniform(-1/self.n_escondida, 1/self.n_escondida, (self.n_saida, self.n_escondida + 1))


    # Função de propagação direta (feedforward) que calcula a saída da rede neural para uma dada entrada e rótulo
    def feedforward(self, x, t): # x é o vetor de entrada e t é o vetor de rótulo (target)
        # Adiciona o bias no início do vetor de entrada (valor fixo 1)
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) # A função insert apenas insere o valor 1 ao início do vetor de entrada

        # Camada escondida (z_in)
        self.z_in = np.dot(self.v, x) # Cada linha de v (pesos de um neurônio da camada escondida) multiplica o vetor de entrada x, a soma dessas multiplicações resulta na entrada da camada escondida (z_in) (produto escalar entre os pesos e as entradas)
        self.z = sigmoide(self.z_in)  # Aplica a função de ativação sigmoide para obter a saída da camada escondida
        self.z = np.insert(self.z, 0, 1)  # Adiciona o bias na saida da camada escondida

        # Camada de saída (y_in)
        self.y_in = np.dot(self.w, self.z) # Cada linha de w (pesos de um neurônio da camada de saída) multiplica o vetor de saída da camada escondida z, a soma dessas multiplicações resulta na entrada da camada de saída (y_in) (produto escalar entre os pesos e a camada escondida)
        self.y = sigmoide(self.y_in) # Aplica a função de ativação sigmoide para obter a saída da camada de saída

        # Calcula o erro da saída em relação ao rótulo (target) e o delta de saída para o backpropagation
        self.erro_saida = (t - self.y) # O erro é a diferença entre o vetor de rótulo (target) e a saída da rede neural
        self.delta_saida = self.erro_saida * derivada_sigmoide(self.y_in) # O delta de saída é o erro multiplicado pela derivada da função de ativação aplicada à entrada da camada de saída (y_in)
        
        # Retorna a saída da camada de saída bruta
        return self.y
    
    # Função de retropropagação (backpropagation) que atualiza os pesos da rede neural com base no erro calculado na função de feedforward
    def backpropagation(self, x):
        # Adiciona o bias no início do vetor de entrada
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) # A função insert apenas insere o valor 1 ao início do vetor de entrada

        # Retropropagação pra camada escondida
        w_sem_bias = self.w[:, 1:] # Remove coluna do bias antes de propagar
        erro_escondida = np.dot(self.delta_saida, w_sem_bias) # O valor de erro da camada escondida é definido como o produto escalar entre o delta de saída e os pesos da camada de saída (sem o bias)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z_in) # O delta de saída é definido como o erro da camada escondida multiplicado pela derivada da função de ativação aplicada à entrada da camada escondida (z_in)

        # Cálculo do termo de correção dos pesos
        delta_w = self.alpha * np.outer(self.delta_saida, self.z) # O delta de atualização dos pesos da camada de saída é calculado como a taxa de aprendizado multiplicada pelo produto vetorial entre o delta de saída e a saída da camada escondida (z)
        delta_v = self.alpha * np.outer(self.delta_escondida, x) # O delta de atualização dos pesos da camada escondida é calculado como a taxa de aprendizado multiplicada pelo produto vetorial entre o delta da camada escondida e a entrada (x)

        # Aplica as correções dos pesos os somando com os termos de correção calculados
        self.w = self.w + delta_w
        self.v = self.v + delta_v

    # Função para recuperar os pesos que obtiveram melhor acurácia durante as validações (útil em casos de parada antecipada, onde os pesos finais podem não ser os melhores)
    def recuperar_pesos(self, melhor_v, melhor_w):
        self.v = melhor_v
        self.w = melhor_w



# Carrega dados e achata imagens (N, H, W) → (N, 120)
caracteristicas_duas_dimensoes = np.load("Conjunto de Dados/caracteres-completo/X.npy") # Carrega o conjunto de dados de características a partir do arquivo X.npy, que contém os valores em formato de matriz 12x10
caracteristicas = caracteristicas_duas_dimensoes.reshape(caracteristicas_duas_dimensoes.shape[0], -1) # Transforma a matriz 12x10 em um vetor de 120 características para cada exemplo
rotulo = np.load("Conjunto de Dados/caracteres-completo/Y_classe.npy") # Carrega o conjunto de dados de rótulos a partir do arquivo Y_classe.npy
caracteristicas_duas_dimensoes_teste_autoral = np.load("Conjunto de Dados/caracteres-completo/X_autoral.npy") # Carrega o conjunto de dados de teste autoral a partir do arquivo X_autoral.npy
caracteristicas_teste_autoral = caracteristicas_duas_dimensoes_teste_autoral.reshape(caracteristicas_duas_dimensoes_teste_autoral.shape[0], -1) # Transforma a matriz 12x10 em um vetor de 120 características para cada exemplo do conjunto de teste autoral

# Divide os dados em conjuntos de treinamento, validação e teste
caracteristicas_treinamento = caracteristicas[0:858] # Os primeiros 858 exemplos são usados para treinamento (33 conjuntos de cada letra)
caracteristicas_validacao = caracteristicas[858:1196] # Os próximos 338 exemplos são usados para validação (13 conjuntos de cada letra)
caracteristicas_teste = caracteristicas[1196:1326] # Os últimos 130 exemplos são usados para teste (5 conjuntos de cada letra)

# Divide os rótulos em conjuntos de treinamento, validação e teste
rotulo_treinamento = rotulo[0:858] # Os primeiros 858 exemplos são usados para treinamento (33 conjuntos de cada letra)
rotulo_validacao = rotulo[858:1196] # Os próximos 338 exemplos são usados para validação (13 conjuntos de cada letra)
rotulo_teste = rotulo[1196:1326] # Os últimos 130 exemplos são usados para teste (5 conjuntos de cada letra)


# Cria pasta de saidas se não existir
os.makedirs("saidas", exist_ok=True)


# Hiperparâmetros da arquitetura e treinamento
camada_escondida = 80 # Define o número de neurônios presentes na camada escondida
taxa_aprendizado = 0.07 # Define a taxa de aprendizado do MLP
paciencia = 18 # Define o número de épocas de paciência para a parada antecipada (número de épocas consecutivas sem melhoria significativa no erro de validação antes de parar o treinamento)
max_epocas = 2000 # Define o número máximo de épocas para o treinamento, caso a parada antecipada não seja acionada

# Inicializa o MLP com os parâmetros definidos
mlp = MLP(120, camada_escondida, 26, taxa_aprendizado)

# Armazena os primeiros pesos (gerados aleatoriamente) como os melhores pesos
melhor_v = mlp.v.copy()
melhor_w = mlp.w.copy()

# Salva os valores dos pesos iniciais
np.savetxt(f"saidas/pesos_iniciais_v.txt", mlp.v, fmt="%.6f")
np.savetxt(f"saidas/pesos_iniciais_w.txt", mlp.w, fmt="%.6f")

menor_erro_quadratico_medio = float('inf') # Inicializa o menor erro quadrático médio com infinito para garantir que o primeiro erro calculado seja considerado como o melhor
erro_quadratico_medio = float('inf')  # Valor utilizado para medição do erro quadrático médio durante o treinamento

parada_antecipada = 0 # Flag de parada antecipada
epocas_sem_melhora = 0 # Contador de épocas sem melhoria no erro de validação

# Contadores de acertos e erros durante a fase de teste
acertos_teste = 0
erros_teste = 0

# Contadores de acertos e erros durante a fase de teste autoral
acertos_teste_autoral = 0
erros_teste_autoral = 0

# Vetor responsável por guardar os erros em cada época
historico_erros = []

i = 0 # Variável para guardar o número de épocas (iterações) durante o treinamento
while (i < max_epocas  and not parada_antecipada): # Condições para o fim do treinamento: atingir 2000 épocas ou acionar a parada antecipada
    i += 1 # Incrementa o número de épocas no começo do loop

    # Treina a rede neural usando o conjunto de treinamento, realizando o feedforward e backpropagation para cada exemplo
    for j in range(len(caracteristicas_treinamento)):
        mlp.feedforward(caracteristicas_treinamento[j], rotulo_treinamento[j])
        mlp.backpropagation(caracteristicas_treinamento[j])

    # Validação da MLP
    erro_quadratico_medio = 0 # Começamos zerando o valor do MSE
    for j in range(len(caracteristicas_validacao)): # Para cada exemplo do conjunto de validação, realizamos o feedforward e somamos o quadrado do erro de saída
        mlp.feedforward(caracteristicas_validacao[j], rotulo_validacao[j])    
        erro_quadratico_medio += np.sum(mlp.erro_saida ** 2) 
    erro_quadratico_medio /= len(caracteristicas_validacao) # Calcula o erro quadrático médio dividindo a soma dos erros quadráticos pelo número de exemplos de validação

    # Verifica se o erro quadrático médio atual é o menor erro quadrático médio registrado até agora. Se for, atualiza os melhores pesos e o menor erro quadrático médio
    if erro_quadratico_medio < menor_erro_quadratico_medio:
        menor_erro_quadratico_medio = erro_quadratico_medio
        melhor_v = mlp.v.copy()
        melhor_w = mlp.w.copy()

    # Registra o erro desta época no histórico
    historico_erros.append(erro_quadratico_medio)

    # Imprime o número da época e o erro quadrático médio para monitorar o progresso do treinamento
    print("Iteracao:", i, "Erro total:", erro_quadratico_medio)

    # Verifica se houve melhoria significativa no erro de validação em relação à época anterior. Se a melhoria for menor que um limiar (0.0001 dividido pela paciência), incrementa o contador de épocas sem melhoria
    if i > 10 and (historico_erros[-2] - erro_quadratico_medio < 0.0001/paciencia):
       epocas_sem_melhora += 1
    else: # Caso contrário, reseta o contador de épocas sem melhoria
        epocas_sem_melhora = 0

    # Se o número de épocas sem melhoria atingir o valor definido pela paciência, aciona a parada antecipada
    if epocas_sem_melhora >= paciencia:
        parada_antecipada = 1

mlp.recuperar_pesos(melhor_v, melhor_w)  # Recupera os melhores pesos encontrados durante o treinamento


# avaliação no teste — matriz 26×26: linha=real, coluna=predito
saidas_brutas_teste = [] # Vetor para armazenar as saídas brutas da rede neural para cada exemplo do conjunto de teste
matriz_confusao = np.zeros((26, 26), dtype=int) # Matriz de confusão para avaliar o desempenho da rede neural no conjunto de teste, onde as linhas representam as classes reais e as colunas representam as classes preditas

# Loop que itera o conjunto de testes
for j in range(len(caracteristicas_teste)):

    resultado = mlp.feedforward(caracteristicas_teste[j], rotulo_teste[j]) # Realiza o feedforward para o exemplo de teste atual e armazena a saída bruta (antes da aplicação do argmax)
    saidas_brutas_teste.append(resultado) # Armazena a saída bruta da rede neural para o exemplo de teste atual no vetor de saídas brutas

    # Calcula a classe predita usando argmax
    classe_predita = argmax(resultado)
    classe_real = argmax(rotulo_teste[j])

    matriz_confusao[classe_real][classe_predita] += 1 # Atualiza a matriz de confusão incrementando a contagem na posição correspondente à classe real e classe predita

    # Verifica se a classe predita é igual à classe real para contar acertos e erros no teste
    if classe_predita == classe_real:
        acertos_teste += 1
    else:
        erros_teste += 1

saidas_brutas_teste_autoral = [] # Vetor para armazenar as saídas brutas da rede neural para cada exemplo do conjunto de teste
matriz_confusao_autoral = np.zeros((26, 26), dtype=int) # Matriz de confusão para avaliar o desempenho da rede neural no conjunto de teste, onde as linhas representam as classes reais e as colunas representam as classes preditas
# Loop que itera o conjunto de testes autoral
for j in range(len(caracteristicas_teste_autoral)):

    resultado = mlp.feedforward(caracteristicas_teste_autoral[j], rotulo[j]) # Realiza o feedforward para o exemplo de teste atual e armazena a saída bruta (antes da aplicação do argmax)
    saidas_brutas_teste_autoral.append(resultado) # Armazena a saída bruta da rede neural para o exemplo de teste atual no vetor de saídas brutas

    # Calcula a classe predita usando argmax
    classe_predita = argmax(resultado)
    classe_real = argmax(rotulo[j])

    matriz_confusao_autoral[classe_real][classe_predita] += 1 # Atualiza a matriz de confusão incrementando a contagem na posição correspondente à classe real e classe predita

    # Verifica se a classe predita é igual à classe real para contar acertos e erros no teste
    if classe_predita == classe_real:
        acertos_teste_autoral += 1
    else:
        erros_teste_autoral += 1


# Imprime os resultados do teste
print("Resultado do teste: ")
print("Acertos no teste:", acertos_teste)
print("Erros no teste:", erros_teste)
print("Acurácia no teste:", acertos_teste / (acertos_teste + erros_teste))
print("Acertos no teste autoral:", acertos_teste_autoral)
print("Erros no teste autoral:", erros_teste_autoral)
print("Acurácia no teste autoral:", acertos_teste_autoral / (acertos_teste_autoral + erros_teste_autoral))
print("Iterações: ", i)


# Salva todos os artefatos em disco
print(f"\n[INFO] Gravando relatorios do experimento em disco...")

with open(f"saidas/hiperparametros_teste.txt", "w") as f:
    f.write("--- Hiperparametros Finais da Arquitetura e Inicializacao ---\n")
    f.write(f"Neuronios de Entrada: {mlp.n_entrada}\n")
    f.write(f"Neuronios na Camada Escondida: {mlp.n_escondida}\n")
    f.write(f"Neuronios de Saida: {mlp.n_saida}\n")
    f.write(f"Taxa de Aprendizado (Alpha): {mlp.alpha}\n")
    f.write(f"Total de Iteracoes executadas: {i}\n")
    f.write(f"Parada Antecipada acionada: {'Sim' if parada_antecipada == 1 else 'Nao'}\n")

np.savetxt(f"saidas/pesos_finais_v.txt", mlp.v, fmt="%.6f")
np.savetxt(f"saidas/pesos_finais_w.txt", mlp.w, fmt="%.6f")
np.savetxt(f"saidas/historico_erros.txt", historico_erros, fmt="%.6f")
np.savetxt(f"saidas/saidas_produzidas.txt", np.array(saidas_brutas_teste), fmt="%.6f")
np.savetxt(f"saidas/matriz_confusao.txt", matriz_confusao, fmt="%d")
np.savetxt(f"saidas/saidas_produzidas_autoral.txt", np.array(saidas_brutas_teste_autoral), fmt="%.6f")
np.savetxt(f"saidas/matriz_confusao_autoral.txt", matriz_confusao_autoral, fmt="%d")

print("\n" + "="*40)
print("[SUCESSO] Todos os artefatos de texto foram gerados na pasta '/saidas'!")
print("="*40)
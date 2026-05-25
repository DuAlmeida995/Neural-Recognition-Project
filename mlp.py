import numpy as np

# funcao de ativacao sigmoide e sua derivada
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)

def derivada_tanh(x_in):
    # f'(x) = 1 - tanh(x)^2
    return 1 - np.tanh(x_in)**2

class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, taxa_aprendizado):
        self.n_entrada = n_entrada
        self.n_escondida = n_escondida
        self.n_saida = n_saida
        self.alpha = taxa_aprendizado
        self.erro_total = 1

        # peso entre camada de entrada e camada escondida (matriz v)
        self.v = np.random.uniform(-1, 1, (self.n_escondida, self.n_entrada + 1)) # o +1 em v e w se deve por conta do bias
        self.v_anterior = np.random.uniform(-1, 1, (self.n_escondida, self.n_entrada + 1)) # o +1 em v e w se deve por conta do bias
        # peso entre camada escondida e camada de saída (matriz w)
        self.w = np.random.uniform(-1, 1, (self.n_saida, self.n_escondida + 1)) 
        self.w_anterior = np.random.uniform(-1, 1, (self.n_saida, self.n_escondida + 1)) 


    def feedforward(self, x):
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) # adiciona o bias na entrada, eu tinha usado a funcao append inicialmente, porem como ela inserir no final, troquei pela insert para inserir o bias no inicio

        # calculo da camada escondida
        self.z = np.dot(self.v, x) # usamos dot para calcular matriz x vetor (tambem podemos usa-lo para calcular matriz x matriz ou vetor x vetor)
        self.a = sigmoide(self.z) 
        self.a = np.insert(self.a, 0, 1) # adiciona o bias na camada escondida

        # calculo da camada de saida
        self.y_in = np.dot(self.w, self.a)

        self.y = sigmoide (self.y_in)

        return self.y
    
    def backpropagation(self, x, t):
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) 

        # calculo do erro na camada de saida
        erro_saida = t - self.y

        self.delta_saida = erro_saida * derivada_sigmoide(self.y_in)
        # self.delta_saida = erro_saida * derivada_tanh(self.y_in)

        self.erro_total = np.sum(erro_saida * erro_saida)/2 # calculo do erro total para monitorar o aprendizado da rede

        # retropropagacao do erro para a camada escondida
        w_sem_bias = self.w[:, 1:] # remove o bias da matriz w para calcular o erro escondido
        erro_escondida = np.dot(self.delta_saida, w_sem_bias)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z)
        delta_w = self.alpha * np.outer(self.delta_saida, self.a) # utilizei o np.outer aqui por conta que ele consegue pegar dois vetores e criar uma matriz onde cada posição (i, j) é o resultado de a[i] X b[j]
        delta_v = self.alpha * np.outer(self.delta_escondida, x)
        self.w = self.w + delta_w
        self.v = self.v + delta_v 



import os  # [MUDANÇA] Requisito para criar o diretório de arquivos de saída

# ... [MANTENHA TODA A SUA CLASSE MLP EXATAMENTE COMO ESTÁ ACIMA] ...

# Carregamento dos dados
caracteristicasDuasDimensoes = np.load("Conjunto de Dados/caracteres-completo/X.npy")
caracteristicas = caracteristicasDuasDimensoes.reshape(caracteristicasDuasDimensoes.shape[0], -1) 

rotulo = np.load("Conjunto de Dados/caracteres-completo/Y_classe.npy")

acertos_treinamento_total = 0
erros_treinamento_total = 0

# --- REQUISITO OBRIGATÓRIO: Garantir criação da pasta de saídas ---
os.makedirs("saidas", exist_ok=True)

for varx in range(1, 2):
    mlp = MLP(120, 65, 26, 0.01)

    # [REQUISITO OBRIGATÓRIO (ii)]: Salvar os pesos iniciais da rede neural
    np.savetxt(f"saidas/pesos_iniciais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")
    np.savetxt(f"saidas/pesos_iniciais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")

    acertos = 0
    acertos_treinamento = 0

    erros = 0
    erros_treinamento = 0

    i = 1
    par_erros_treinamento = 1
    erro_aceitavel = 0
    
    # [REQUISITO OBRIGATÓRIO (iv)]: Lista para persistir o histórico de erros cometidos
    historico_erros = []

    # Enquanto os pesos não convergirem (ajustado com .copy() para isolar endereços de memória)
    while ((not (np.array_equal(mlp.w, mlp.w_anterior) and np.array_equal(mlp.v, mlp.v_anterior))) and i < 10000 and erro_aceitavel != 1): 
        mlp.v_anterior = mlp.v.copy()
        mlp.w_anterior = mlp.w.copy()
        
        par_erros_treinamento = 0
        for j in range(len(caracteristicas)-130):
            mlp.feedforward(caracteristicas[j])
            mlp.backpropagation(caracteristicas[j], rotulo[j])
            par_erros_treinamento += mlp.erro_total
            
        par_erros_treinamento = par_erros_treinamento / (len(caracteristicas)-130) 

        # Armazena o erro médio computado nesta iteração do treinamento
        historico_erros.append(par_erros_treinamento)

        i += 1

        print("Iteracao:", i, "Erro total:", par_erros_treinamento)

        # Critério de Parada Antecipada implementado pelo grupo
        if i >= 300 and par_erros_treinamento < 0.05:
            erro_aceitavel = 1

    # Finalizado o treino, extrai os acertos no conjunto de treino
    for j in range(len(caracteristicas)-130):
        resultado = mlp.feedforward(caracteristicas[j])
        if np.argmax(resultado) == np.argmax(rotulo[j]):
            acertos += 1
        else:
            erros += 1
            
    # Avaliação no conjunto de teste (últimas 130 instâncias) + Matriz de Confusão
    saidas_brutas_teste = []
    matriz_confusao = np.zeros((26, 26), dtype=int)  # 26 classes correspondentes às letras

    for j in range(len(caracteristicas)-130, len(caracteristicas)):
        resultado = mlp.feedforward(caracteristicas[j])
        saidas_brutas_teste.append(resultado)  # Coleta as saídas brutas da rede de teste
        
        classe_predita = np.argmax(resultado)
        classe_real = np.argmax(rotulo[j])
        
        # Mapeamento da matriz: Linha (Classe Real) vs Coluna (Classe Predita)
        matriz_confusao[classe_real][classe_predita] += 1
        
        if classe_predita == classe_real:
            acertos += 1
            acertos_treinamento += 1
        else:
            erros += 1
            erros_treinamento += 1

    print("Resultado do teste: ", varx)
    print("Acertos:", acertos)
    print("Erros:", erros)
    print("Acertos no treinamento:", acertos_treinamento)
    print("Erros no treinamento:", erros_treinamento)
    print("Acurácia no treinamento:", acertos_treinamento / (acertos_treinamento + erros_treinamento))
    print("Iterações: ", i)

    acertos_treinamento_total += acertos_treinamento
    erros_treinamento_total += erros_treinamento

    # =====================================================================
    # EXPORTAÇÃO DOS ARQUIVOS DE TEXTO REQUISITADOS (TXT / LEGÍVEIS)
    # =====================================================================
    print(f"\n[INFO] Gravando relatorios do experimento {varx} em disco...")

    # (i) Arquivo de hiperparâmetros finais e de inicialização
    with open(f"saidas/hiperparametros_teste_{varx}.txt", "w") as f:
        f.write("--- Hiperparametros Finais da Arquitetura e Inicializacao ---\n")
        f.write(f"Neuronios de Entrada: {mlp.n_entrada}\n")
        f.write(f"Neuronios na Camada Escondida: {mlp.n_escondida}\n")
        f.write(f"Neuronios de Saida: {mlp.n_saida}\n")
        f.write(f"Taxa de Aprendizado (Alpha): {mlp.alpha}\n")
        f.write(f"Total de Iteracoes executadas: {i - 1}\n")
        f.write(f"Parada Antecipada acionada: {'Sim' if erro_aceitavel == 1 else 'Nao'}\n")

    # (iii) Arquivo contendo os pesos finais da rede neural
    np.savetxt(f"saidas/pesos_finais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")
    np.savetxt(f"saidas/pesos_finais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")

    # (iv) Arquivo contendo o erro cometido pela rede em cada iteração
    np.savetxt(f"saidas/historico_erros_teste_{varx}.txt", historico_erros, fmt="%.6f")

    # (v) Arquivo contendo as saídas brutas produzidas para cada dado de teste
    np.savetxt(f"saidas/saidas_produzidas_teste_{varx}.txt", np.array(saidas_brutas_teste), fmt="%.6f")

    # [REQUISITO EXIGIDO PARA O VÍDEO]: Matriz de Confusão salva em formato texto plano
    np.savetxt(f"saidas/matriz_confusao_teste_{varx}.txt", matriz_confusao, fmt="%d")

print("\n" + "="*40)
print("Acertos totais no treinamento:", acertos_treinamento_total)
print("Erros totais no treinamento:", erros_treinamento_total)
print("Acurácia total no treinamento:", acertos_treinamento_total / (acertos_treinamento_total + erros_treinamento_total))
print("[SUCESSO] Todos os artefatos de texto foram gerados na pasta '/saidas'!")
print("="*40)
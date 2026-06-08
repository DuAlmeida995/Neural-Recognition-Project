# MEMBROS DO GRUPO:
# André Portela Lino - 15634885
# Davi Lima de Oliveira - 15648741
# Eduardo Almeida Cavalcanti de Melo - 15526004
# Eric Isin Wang Chou - 15574579
# Júlio Arroio Silva - 15466241
# Karina Yang Chen - 15466658
# TURMA 94



# MLP para reconhecimento de caracteres A-Z
# Arquitetura: 120 entrada → 65 escondida → 26 saída

import numpy as np

# funções de ativação e derivadas — usadas no feedforward e backprop

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)

def derivada_tanh(x_in):
    # f'(x) = 1 - tanh(x)^2
    return 1 - np.tanh(x_in)**2

# aprox. universal com uma camada escondida — dá conta das 26 letras
class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, taxa_aprendizado):
        self.n_entrada = n_entrada
        self.n_escondida = n_escondida
        self.n_saida = n_saida
        self.alpha = taxa_aprendizado    # passo da descida do gradiente
        self.erro_total = 1              # > 0 pra entrar no while

        # uniforme(-1,1) — pesos iguais travam o aprendizado (simetria)
        self.v = np.random.uniform(-0.03, 0.03, (self.n_escondida, self.n_entrada + 1))  # +1 = bias
        self.v_anterior = np.random.uniform(-0.03, 0.03, (self.n_escondida, self.n_entrada + 1))
        self.w = np.random.uniform(-0.02, 0.02, (self.n_saida, self.n_escondida + 1))    # +1 = bias
        self.w_anterior = np.random.uniform(-0.02, 0.02, (self.n_saida, self.n_escondida + 1))


    # -------------------------------------------------------------------------
    # Propagação Direta (Feedforward)
    # -------------------------------------------------------------------------
    # O sinal percorre a rede da entrada para a saída, camada por camada.
    # Para cada neurônio: calcula-se a combinação linear de entradas e
    # pesos e aplica-se a função de ativação para obter a saída do neurônio.
    def feedforward(self, x, t):
        # BIAS no início do vetor de entrada (valor fixo 1) - utilizado o insert para colocar no início
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1)

        # Camada escondida 
        # cada linha de V (pesos de um neurônio) multiplica o vetor x e soma os resultados → um único valor.
        self.z = np.dot(self.v, x)
        self.a = sigmoide(self.z)  # guardamos pra reusar no backprop
        self.a = np.insert(self.a, 0, 1)  # bias da escondida

        # camada de saída — y_in guardado pro backprop
        self.y_in = np.dot(self.w, self.a)
        self.y = sigmoide(self.y_in)

        # delta da saída — regra delta
        erro_saida = t - self.y
        self.delta_saida = erro_saida * derivada_sigmoide(self.y_in)
        # self.delta_saida = erro_saida * derivada_tanh(self.y_in)  # versão com tanh
        self.erro_total = np.sum(erro_saida * erro_saida) / 2  # só pra monitorar

        return self.y
    
    def backpropagation(self, x):
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1)  # bias na entrada, igual ao feedforward

        # retroprop pra escondida — remove col do bias antes de propagar
        w_sem_bias = self.w[:, 1:]
        erro_escondida = np.dot(self.delta_saida, w_sem_bias)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z)

        # outer: cada pos (i,j) = delta[i] * entrada[j]
        delta_w = self.alpha * np.outer(self.delta_saida, self.a)
        delta_v = self.alpha * np.outer(self.delta_escondida, x)

        # Aplica as correções: w_novo = w_antigo + ΔW
        self.w = self.w + delta_w
        self.v = self.v + delta_v

    def recuperar_pesos(self, melhorv, melhorw):
        self.v = melhorv
        self.w = melhorw



import os

# carrega dados e achata imagens (N, H, W) → (N, 120)
caracteristicasDuasDimensoes = np.load("Conjunto de Dados/caracteres-completo/X.npy")
caracteristicas = caracteristicasDuasDimensoes.reshape(caracteristicasDuasDimensoes.shape[0], -1)

caracteristicasTreinamento = caracteristicas[0:858]
caracteristicasValidacao = caracteristicas[858:858+338]
caracteristicasTeste = caracteristicas[858+338:858+338+130]


rotulo = np.load("Conjunto de Dados/caracteres-completo/Y_classe.npy")

rotuloTreinamento = rotulo[0:858]
rotuloValidacao = rotulo[858:858+338]
rotuloTeste = rotulo[858+338:858+338+130]

acertos_treinamento_total = 0
erros_treinamento_total = 0

os.makedirs("saidas", exist_ok=True)  # cria pasta se não existir

for varx in range(1, 2):
    mlp = MLP(120, 50, 26, 0.07)  # 120 → 65 → 26, α=0.01

    # salva pesos iniciais pra comparar depois
    np.savetxt(f"saidas/pesos_iniciais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")
    np.savetxt(f"saidas/pesos_iniciais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")

    acertos = 0
    acertos_treinamento = 0
    erros = 0
    erros_treinamento = 0

    i = 1
    par_erros_treinamento = 1  # > 0 pra entrar no while
    erro_aceitavel = 0         # flag de parada antecipada

    melhorV = mlp.v.copy()  # pra armazenar os melhores pesos encontrados
    melhorW = mlp.w.copy()
    melhor_erro = float('inf')  # inicializa com infinito pra garantir que o primeiro erro seja melhor

    historico_erros = []

    # treina até convergir, atingir 10000 épocas ou parada antecipada
    # últimos 130 exemplos são reservados pra teste
    while ((not (np.array_equal(mlp.w, mlp.w_anterior) and np.array_equal(mlp.v, mlp.v_anterior))) and erro_aceitavel < 10):
        mlp.v_anterior = mlp.v.copy()
        mlp.w_anterior = mlp.w.copy()

        for j in range(len(caracteristicasTreinamento)):
            mlp.feedforward(caracteristicasTreinamento[j], rotuloTreinamento[j])
            mlp.backpropagation(caracteristicasTreinamento[j])

        # treinamento online: um exemplo por vez

        par_erros_treinamento = 0
        for j in range(len(caracteristicasValidacao)):
            mlp.feedforward(caracteristicasValidacao[j], rotuloValidacao[j])    
            par_erros_treinamento += mlp.erro_total

        # erro médio da época
        par_erros_treinamento = par_erros_treinamento / (len(caracteristicasValidacao))
        if par_erros_treinamento < melhor_erro:
            melhor_erro = par_erros_treinamento
            melhorV = mlp.v.copy()
            melhorW = mlp.w.copy()

        # Registra o erro desta época no histórico
        historico_erros.append(par_erros_treinamento)

        i += 1

        print("Iteracao:", i, "Erro total:", par_erros_treinamento)

        # parada antecipada: aguarda 300 épocas antes pra rede ter tempo de aprender
        if i > 10 and (historico_erros[-2] - par_erros_treinamento < 0.000001):
            erro_aceitavel += 1
        else:
            erro_aceitavel = 0


    # avaliação no teste — matriz 26×26: linha=real, coluna=predito
    saidas_brutas_teste = []
    matriz_confusao = np.zeros((26, 26), dtype=int)
    mlp.recuperar_pesos(melhorV, melhorW)  # usa os melhores pesos encontrados durante o treinamento

    for j in range(len(caracteristicasTeste)):
        # troca por teste_autoral[j] pra usar os dados autorais modificados
        teste_autoralX = np.load("Conjunto de Dados/caracteres-completo/X_autoral.npy")
        
        #resultado = mlp.feedforward(teste_autoral[j])
        resultado = mlp.feedforward(caracteristicasTeste[j], rotuloTeste[j])
        saidas_brutas_teste.append(resultado)

        classe_predita = np.argmax(resultado)
        classe_real = np.argmax(rotuloTeste[j])
        matriz_confusao[classe_real][classe_predita] += 1

        if classe_predita == classe_real:
            acertos_treinamento += 1
        else:
            erros_treinamento += 1

    print("Resultado do teste: ", varx)
    print("Acertos no treinamento:", acertos_treinamento)
    print("Erros no treinamento:", erros_treinamento)
    print("Acurácia no treinamento:", acertos_treinamento / (acertos_treinamento + erros_treinamento))
    print("Iterações: ", i)

    acertos_treinamento_total += acertos_treinamento
    erros_treinamento_total += erros_treinamento

    # salva todos os artefatos em disco
    print(f"\n[INFO] Gravando relatorios do experimento {varx} em disco...")

    with open(f"saidas/hiperparametros_teste_{varx}.txt", "w") as f:
        f.write("--- Hiperparametros Finais da Arquitetura e Inicializacao ---\n")
        f.write(f"Neuronios de Entrada: {mlp.n_entrada}\n")
        f.write(f"Neuronios na Camada Escondida: {mlp.n_escondida}\n")
        f.write(f"Neuronios de Saida: {mlp.n_saida}\n")
        f.write(f"Taxa de Aprendizado (Alpha): {mlp.alpha}\n")
        f.write(f"Total de Iteracoes executadas: {i - 1}\n")
        f.write(f"Parada Antecipada acionada: {'Sim' if erro_aceitavel == 1 else 'Nao'}\n")

    np.savetxt(f"saidas/pesos_finais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")  # pesos finais — dá pra recarregar sem retreinar
    np.savetxt(f"saidas/pesos_finais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")
    np.savetxt(f"saidas/historico_erros_teste_{varx}.txt", historico_erros, fmt="%.6f")  # curva de aprendizado
    np.savetxt(f"saidas/saidas_produzidas_teste_{varx}.txt", np.array(saidas_brutas_teste), fmt="%.6f")  # saídas brutas (antes do argmax)
    np.savetxt(f"saidas/matriz_confusao_teste_{varx}.txt", matriz_confusao, fmt="%d")

print("\n" + "="*40)
print("Acertos totais no treinamento:", acertos_treinamento_total)
print("Erros totais no treinamento:", erros_treinamento_total)
print("Acurácia total no treinamento:", acertos_treinamento_total / (acertos_treinamento_total + erros_treinamento_total))
print("[SUCESSO] Todos os artefatos de texto foram gerados na pasta '/saidas'!")
print("="*40)
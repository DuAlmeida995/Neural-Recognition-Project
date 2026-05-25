# =============================================================================
# Rede Neural MLP (Multilayer Perceptron) para Reconhecimento de Caracteres
# =============================================================================
# Este arquivo implementa do zero uma rede neural Perceptron Multicamada (MLP),
# seguindo os conceitos estudados em aula: camadas de entrada, escondida e saída,
# propagação direta (feedforward), retropropagação do erro (backpropagation)
# e a Regra Delta Generalizada para ajuste dos pesos.
#
# Arquitetura utilizada:
#   - Camada de entrada : 120 neurônios (vetor de pixels de cada imagem de caractere)
#   - Camada escondida  : 65 neurônios (com bias)
#   - Camada de saída   : 26 neurônios (um para cada letra do alfabeto)
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# Funções de Ativação e suas Derivadas
# -----------------------------------------------------------------------------
# As funções de ativação introduzem não-linearidade na rede, permitindo que a
# MLP aprenda fronteiras de decisão complexas — algo impossível para um
# Perceptron simples de uma única camada.
#
# A derivada de cada função é necessária no backpropagation para calcular o
# gradiente do erro em relação aos pesos (regra da cadeia).

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)

def derivada_tanh(x_in):
    # f'(x) = 1 - tanh(x)^2
    return 1 - np.tanh(x_in)**2

# -----------------------------------------------------------------------------
# Classe MLP — Perceptron Multicamada
# -----------------------------------------------------------------------------
# A MLP é uma extensão direta do Perceptron simples estudado em aula:
# enquanto o Perceptron tem apenas uma camada de pesos, a MLP adiciona uma
# ou mais camadas intermediárias (escondidas), permitindo aprender padrões
# não-linearmente separáveis — como o reconhecimento das 26 letras do alfabeto.
#
# Aqui usamos uma única camada escondida, o que já é suficiente para ser
# um aproximador universal de funções.
class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, taxa_aprendizado):
        # Guarda os parâmetros que definem a arquitetura da rede
        self.n_entrada = n_entrada       # número de neurônios na camada de entrada
        self.n_escondida = n_escondida   # número de neurônios na camada escondida
        self.n_saida = n_saida           # número de neurônios na camada de saída
        self.alpha = taxa_aprendizado    # taxa de aprendizado (α): controla o
                                         # tamanho do passo na descida do gradiente.
                                         # α grande = aprende rápido mas pode oscilar;
                                         # α pequeno = mais estável mas converge devagar.
        self.erro_total = 1              # inicializado com valor > 0 para entrar no loop

        # Inicialização ALEATÓRIA dos pesos: valores uniformemente distribuídos
        # entre -1 e 1. Isso é fundamental — pesos iguais fariam todos os
        # neurônios aprender exatamente a mesma coisa (problema de simetria).
        #
        # Matriz V: pesos entre a camada de ENTRADA e a camada ESCONDIDA.
        # Dimensão: (n_escondida × (n_entrada + 1))
        # O +1 corresponde ao peso do BIAS — um neurônio extra de valor fixo 1
        # que permite deslocar a função de ativação, dando mais liberdade ao modelo.
        self.v = np.random.uniform(-1, 1, (self.n_escondida, self.n_entrada + 1))
        self.v_anterior = np.random.uniform(-1, 1, (self.n_escondida, self.n_entrada + 1)) # cópia para verificar convergência
        # Matriz W: pesos entre a camada ESCONDIDA e a camada de SAÍDA.
        # Dimensão: (n_saida × (n_escondida + 1)) — o +1 é o bias da camada escondida.
        self.w = np.random.uniform(-1, 1, (self.n_saida, self.n_escondida + 1))
        self.w_anterior = np.random.uniform(-1, 1, (self.n_saida, self.n_escondida + 1)) # cópia para verificar convergência


    # -------------------------------------------------------------------------
    # Propagação Direta (Feedforward)
    # -------------------------------------------------------------------------
    # O sinal percorre a rede da entrada para a saída, camada por camada.
    # Para cada neurônio: calcula-se a combinação linear de entradas e
    # pesos e aplica-se a função de ativação para obter a saída do neurônio.
    def feedforward(self, x):
        # Insere o BIAS no início do vetor de entrada (valor fixo = 1).
        # Usamos np.insert(..., 0, 1) para colocar o 1 na posição 0 do vetor
        # (início), pois np.append colocaria no final — o que quebraria a ordem.
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1)

        # --- Camada Escondida ---
        # np.dot calcula o produto matriz-vetor: cada linha de V (pesos de um
        # neurônio) multiplica o vetor x e soma os resultados → um único valor.
        self.z = np.dot(self.v, x)

        # a = sigmoide(z)  →  saída de cada neurônio escondido após a ativação.
        # Guardamos 'a' como atributo pois ele será reutilizado no backpropagation.
        self.a = sigmoide(self.z)

        # Insere o bias da camada escondida antes de passar para a camada de saída.
        self.a = np.insert(self.a, 0, 1)

        # --- Camada de Saída ---
        # y_in = W · a  → cálculo para cada neurônio de saída.
        # Guardamos 'y_in' pois ele é necessário para calcular a derivada no backprop.
        self.y_in = np.dot(self.w, self.a)

        # y = sigmoide(y_in)  →  saída final da rede.
        # Cada um dos 26 valores representa o "grau de confiança" da rede de que
        # o padrão de entrada pertence àquela classe (letra do alfabeto).
        self.y = sigmoide(self.y_in)

        return self.y
    
    # -------------------------------------------------------------------------
    # Retropropagação do Erro (Backpropagation)
    # -------------------------------------------------------------------------
    # Após o feedforward, o erro é propagado de volta pela rede, da saída para
    # a entrada. Usando a regra da cadeia do cálculo, determinamos o quanto
    # cada peso contribuiu para o erro e o ajustamos na direção que minimiza
    # esse erro (descida do gradiente). 
    #
    # Parâmetros:
    #   x — entrada do exemplo de treinamento
    #   t — saída desejada / rótulo (target), no formato one-hot
    def backpropagation(self, x, t):
        # Reinsere o bias no vetor de entrada para ficar consistente com o feedforward
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1)

        # --- Erro na Camada de Saída ---
        # Diferença entre o que era esperado (t) e o que a rede produziu (y).
        erro_saida = t - self.y

        # Delta da camada de saída: aplica a Regra Delta.
        # δ_saída = erro_saída × f'(y_in)
        self.delta_saida = erro_saida * derivada_sigmoide(self.y_in)
        # self.delta_saida = erro_saida * derivada_tanh(self.y_in)  # versão com tanh

        # Erro total do exemplo: E = Σ(t - y)² / 2
        # Usamos a metade do quadrado para que a derivada seja simplesmente (t - y).
        # Serve apenas para monitorar o progresso do treinamento — não altera os pesos.
        self.erro_total = np.sum(erro_saida * erro_saida) / 2

        # --- Retropropagação para a Camada Escondida ---
        # O erro "chega" aos neurônios escondidos através dos pesos W.
        # Removemos a coluna do bias de W (a coluna 0) porque o neurônio de bias
        # não existe na camada anterior, portanto não recebe gradiente.
        w_sem_bias = self.w[:, 1:]
        erro_escondida = np.dot(self.delta_saida, w_sem_bias)

        # Delta da camada escondida: mesma Regra Delta, mas agora o erro é o
        # sinal retropropagado pelos pesos W a partir da camada de saída.
        # δ_escondida = erro_escondida × f'(z)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z)

        # --- Atualização dos Pesos (Regra Delta Generalizada) ---
        # ΔW = α × (δ_saída ⊗ a)  → produto externo: gera a matriz de correções de W
        # ΔV = α × (δ_escondida ⊗ x)  → produto externo: gera a matriz de correções de V
        # np.outer(a, b) cria uma matriz onde a posição (i, j) = a[i] × b[j],
        # calculando de uma só vez a atualização de todos os pesos da camada.
        delta_w = self.alpha * np.outer(self.delta_saida, self.a)
        delta_v = self.alpha * np.outer(self.delta_escondida, x)

        # Aplica as correções: w_novo = w_antigo + ΔW
        self.w = self.w + delta_w
        self.v = self.v + delta_v



import os  # necessário para criar o diretório de saída caso ele ainda não exista

# =============================================================================
# Script Principal — Treinamento e Avaliação da MLP
# =============================================================================

# -----------------------------------------------------------------------------
# Carregamento dos dados de entrada e rótulos
# -----------------------------------------------------------------------------
# O conjunto de dados contém imagens binarizadas de caracteres (letras A–Z).
# Cada imagem é uma matriz de pixels que precisamos "achatar" em um vetor 1D
# (vetor de características), pois a MLP recebe vetores como entrada, não
# matrizes 2D.
#
# X.npy       → shape (N, H, W): N imagens de altura H e largura W
# Y_classe.npy→ rótulos one-hot: vetor de 26 posições, onde só a posição da
#               letra correta vale 1 e todas as demais valem 0.
caracteristicasDuasDimensoes = np.load("Conjunto de Dados/caracteres-completo/X.npy")
caracteristicas = caracteristicasDuasDimensoes.reshape(caracteristicasDuasDimensoes.shape[0], -1)
# Após o reshape: shape (N, 120) — cada linha é o vetor de 120 pixels de uma imagem

rotulo = np.load("Conjunto de Dados/caracteres-completo/Y_classe.npy")

# Acumuladores para o caso de múltiplos experimentos (o loop roda de 1 a 1 aqui)
acertos_treinamento_total = 0
erros_treinamento_total = 0

# Garante que a pasta de saída existe antes de gravar qualquer arquivo
os.makedirs("saidas", exist_ok=True)

# O loop externo permite repetir o experimento com índice de controle (varx),
# útil para salvar os arquivos de saída com nomes distintos por experimento.
for varx in range(1, 2):
    # Instancia a MLP com a arquitetura e taxa de aprendizado definidas.
    # 120 entradas → 65 neurônios escondidos → 26 saídas, com α = 0.01
    mlp = MLP(120, 65, 26, 0.01)

    # Salva os pesos ANTES do treinamento para comparar com os pesos finais
    # e analisar a evolução do aprendizado.
    np.savetxt(f"saidas/pesos_iniciais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")
    np.savetxt(f"saidas/pesos_iniciais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")

    acertos = 0              # acertos totais (treino + teste)
    acertos_treinamento = 0  # acertos apenas no conjunto de teste

    erros = 0
    erros_treinamento = 0

    i = 1                    # contador de épocas
    par_erros_treinamento = 1  # erro médio da época (> 0 para entrar no while)
    erro_aceitavel = 0         # flag de Parada Antecipada (0=continua, 1=para)

    # Histórico do erro médio por época, usado para plotar a curva de aprendizado
    historico_erros = []

    # -------------------------------------------------------------------------
    # Loop de Treinamento (por épocas)
    # -------------------------------------------------------------------------
    # Uma "época" = uma passagem completa por todos os exemplos de treinamento
    # (os primeiros len-130; os últimos 130 são reservados para teste).
    #
    # O treinamento para quando qualquer uma das condições é verdadeira:
    #   1. Convergência: os pesos não mudam mais entre duas épocas consecutivas
    #   2. Limite de épocas atingido (10.000)
    #   3. Critério de Parada Antecipada satisfeito
    while ((not (np.array_equal(mlp.w, mlp.w_anterior) and np.array_equal(mlp.v, mlp.v_anterior))) and i < 10000 and erro_aceitavel != 1):
        # Cópia dos pesos atuais para comparar com os novos pesos no próximo ciclo
        mlp.v_anterior = mlp.v.copy()
        mlp.w_anterior = mlp.w.copy()

        # Treinamento online (estocástico): apresenta um exemplo por vez,
        # ajustando os pesos após cada exemplo (feedforward + backpropagation).
        par_erros_treinamento = 0
        for j in range(len(caracteristicas)-130):
            mlp.feedforward(caracteristicas[j])
            mlp.backpropagation(caracteristicas[j], rotulo[j])
            par_erros_treinamento += mlp.erro_total

        # Erro médio da época: soma de todos os erros dividida pelo nº de exemplos
        par_erros_treinamento = par_erros_treinamento / (len(caracteristicas)-130)

        # Registra o erro desta época no histórico
        historico_erros.append(par_erros_treinamento)

        i += 1

        print("Iteracao:", i, "Erro total:", par_erros_treinamento)

        # --- Critério de Parada Antecipada ---
        # Inspirado em Haykin: interrompe o treinamento quando o erro já está
        # baixo o suficiente, evitando que a rede "decore" os dados de treino
        # (overfitting) e perca a capacidade de generalizar.
        # Só verificamos após 300 épocas para garantir que a rede teve tempo
        # de aprender antes de parar.
        if i >= 300 and par_erros_treinamento < 0.05:
            erro_aceitavel = 1

    # -------------------------------------------------------------------------
    # Avaliação no Conjunto de Treinamento
    # -------------------------------------------------------------------------
    # Após o treinamento, passamos os mesmos exemplos de treino pela rede
    # (sem ajustar pesos) para verificar quantos ela classifica corretamente.
    # A classe predita é o índice do maior valor da saída (argmax) — ou seja,
    # a letra com maior "confiança" da rede. O rótulo correto também é one-hot,
    # então argmax retorna o índice onde vale 1 (a letra verdadeira).
    for j in range(len(caracteristicas)-130):
        resultado = mlp.feedforward(caracteristicas[j])
        if np.argmax(resultado) == np.argmax(rotulo[j]):
            acertos += 1
        else:
            erros += 1

    # -------------------------------------------------------------------------
    # Avaliação no Conjunto de Teste + Matriz de Confusão
    # -------------------------------------------------------------------------
    # Os últimos 130 exemplos NUNCA foram vistos durante o treinamento — são
    # o conjunto de TESTE. Avaliá-los separadamente permite estimar a capacidade
    # de generalização da rede (desempenho em dados novos).
    #
    # A Matriz de Confusão é uma tabela 26×26 que resume os resultados:
    #   - Linha i   → classe REAL  (letra verdadeira do exemplo)
    #   - Coluna j  → classe PREDITA (letra que a rede classificou)
    # A diagonal principal são acertos; fora dela são erros, mostrando
    # com qual outra letra a rede confundiu cada padrão.
    saidas_brutas_teste = []
    matriz_confusao = np.zeros((26, 26), dtype=int)  # 26 classes = letras A a Z

    for j in range(len(caracteristicas)-130, len(caracteristicas)):
        resultado = mlp.feedforward(caracteristicas[j])
        saidas_brutas_teste.append(resultado)  # guarda a saída bruta (antes do argmax)

        # Determina qual letra a rede prediz e qual era a letra correta
        classe_predita = np.argmax(resultado)
        classe_real = np.argmax(rotulo[j])

        # Registra na matriz: linha = real, coluna = predito
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

    # =========================================================================
    # Exportação dos Resultados para Arquivos de Texto
    # =========================================================================
    # Todos os artefatos relevantes são salvos em disco para análise posterior,
    # reprodutibilidade e comparação entre diferentes configurações.
    print(f"\n[INFO] Gravando relatorios do experimento {varx} em disco...")

    # Resumo da arquitetura e da execução: hiperparâmetros usados, total de
    # épocas e se o critério de Parada Antecipada foi acionado
    with open(f"saidas/hiperparametros_teste_{varx}.txt", "w") as f:
        f.write("--- Hiperparametros Finais da Arquitetura e Inicializacao ---\n")
        f.write(f"Neuronios de Entrada: {mlp.n_entrada}\n")
        f.write(f"Neuronios na Camada Escondida: {mlp.n_escondida}\n")
        f.write(f"Neuronios de Saida: {mlp.n_saida}\n")
        f.write(f"Taxa de Aprendizado (Alpha): {mlp.alpha}\n")
        f.write(f"Total de Iteracoes executadas: {i - 1}\n")
        f.write(f"Parada Antecipada acionada: {'Sim' if erro_aceitavel == 1 else 'Nao'}\n")

    # Pesos FINAIS após o treinamento: permitem recarregar a rede treinada
    # sem precisar retreinar (basta carregar os pesos e usar o feedforward)
    np.savetxt(f"saidas/pesos_finais_v_teste_{varx}.txt", mlp.v, fmt="%.6f")
    np.savetxt(f"saidas/pesos_finais_w_teste_{varx}.txt", mlp.w, fmt="%.6f")

    # Histórico do erro médio por época: com ele podemos plotar a curva de
    # aprendizado e verificar se a rede convergiu ou ficou oscilando
    np.savetxt(f"saidas/historico_erros_teste_{varx}.txt", historico_erros, fmt="%.6f")

    # Saídas brutas da rede para os exemplos de teste (vetor de 26 valores antes
    # do argmax): útil para analisar o nível de confiança da rede em cada predição
    np.savetxt(f"saidas/saidas_produzidas_teste_{varx}.txt", np.array(saidas_brutas_teste), fmt="%.6f")

    # Matriz de confusão 26×26 em formato texto plano para análise dos erros
    np.savetxt(f"saidas/matriz_confusao_teste_{varx}.txt", matriz_confusao, fmt="%d")

print("\n" + "="*40)
print("Acertos totais no treinamento:", acertos_treinamento_total)
print("Erros totais no treinamento:", erros_treinamento_total)
print("Acurácia total no treinamento:", acertos_treinamento_total / (acertos_treinamento_total + erros_treinamento_total))
print("[SUCESSO] Todos os artefatos de texto foram gerados na pasta '/saidas'!")
print("="*40)
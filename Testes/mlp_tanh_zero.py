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

        # peso entre camada de entrada e camada escondida (matriz v)
        self.v = np.zeros((self.n_escondida, self.n_entrada + 1))+0.01 # o +1 em v e w se deve por conta do bias
        self.v_anterior = np.zeros((self.n_escondida, self.n_entrada + 1)) # o +1 em v e w se deve por conta do bias
        # peso entre camada escondida e camada de saída (matriz w)
        self.w = np.zeros((self.n_saida, self.n_escondida + 1))+0.01
        self.w_anterior = np.zeros((self.n_saida, self.n_escondida + 1))

    def feedforward(self, x):
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) # adiciona o bias na entrada, eu tinha usado a funcao append inicialmente, porem como ela inserir no final, troquei pela insert para inserir o bias no inicio

        # calculo da camada escondida
        self.z = np.dot(self.v, x) # usamos dot para calcular matriz x vetor (tambem podemos usa-lo para calcular matriz x matriz ou vetor x vetor)
        self.a = sigmoide(self.z) 
        self.a = np.insert(self.a, 0, 1) # adiciona o bias na camada escondida

        # calculo da camada de saida
        self.y_in = np.dot(self.w, self.a)
        self.y = tanh(self.y_in)
        # self.y = np.heaviside(tanh(self.y_in) - 0.5, 0) #return np.heaviside(self.y - 0.5 ,0) # a funcao heaviside é equivalente a função step, ela é usada para converter a saida da rede em 0 ou 1, dependendo se o valor é menor ou maior que 0.5
        return self.y
    
    def backpropagation(self, x, t):
        if len(x) == self.n_entrada:
            x = np.insert(x, 0, 1) 

        # calculo do erro na camada de saida
        erro_saida = t - self.y
        self.delta_saida = erro_saida * derivada_tanh(self.y_in)
        self.erro_total = np.sum(erro_saida ** 2)/2 # calculo do erro total para monitorar o aprendizado da rede

        # retropropagacao do erro para a camada escondida
        w_sem_bias = self.w[:, 1:] # remove o bias da matriz w para calcular o erro escondido
        erro_escondida = np.dot(self.delta_saida, w_sem_bias)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z)
        delta_w = self.alpha * np.outer(self.delta_saida, self.a) # utilizei o np.outer aqui por conta que ele consegue pegar dois vetores e criar uma matriz onde cada posição (i, j) é o resultado de a[i] X b[j]
        delta_v = self.alpha * np.outer(self.delta_escondida, x)
        self.w = self.w + delta_w
        self.v = self.v + delta_v 

# # Teste da rede neural para aprender a função XOR
# mlp = MLP(120, 100, 26, 0.1);

# i = 1

# while not (np.array_equal(mlp.w, mlp.w_anterior) and np.array_equal(mlp.v, mlp.v_anterior)): # Enquanto os pesos não convergirem
#     mlp.v_anterior = mlp.v
#     mlp.w_anterior = mlp.w

#     mlp.feedforward(np.array([-1,-1]))
#     mlp.backpropagation(np.array([-1,-1]), np.array([0]))

#     mlp.feedforward(np.array([-1,1]))
#     mlp.backpropagation(np.array([-1,1]), np.array([1]))

#     mlp.feedforward(np.array([1,-1]))
#     mlp.backpropagation(np.array([1,-1]), np.array([1]))

#     mlp.feedforward(np.array([1,1]))
#     mlp.backpropagation(np.array([1,1]), np.array([0]))

#     print("Iteracao:", i, "Pesos v:", mlp.v, "Pesos w:", mlp.w, "Erro total:", mlp.erro_total)
#     i += 1

# print("Saida para [-1, -1]:", mlp.feedforward(np.array([-1, -1])))
# print("Saida para [-1, 1]:", mlp.feedforward(np.array([-1, 1])))
# print("Saida para [1, -1]:", mlp.feedforward(np.array([1, -1])))
# print("Saida para [1, 1]:", mlp.feedforward(np.array([1, 1])))



# Carregamento dos dados
caracteristicasDuasDimensoes = np.load("Conjunto de Dados/caracteres-completo/X.npy")
caracteristicas = caracteristicasDuasDimensoes.reshape(caracteristicasDuasDimensoes.shape[0], -1) # reshape para transformar a matriz de 3 dimensoes em uma matriz de 2 dimensoes, onde cada linha é um vetor de caracteristicas
# print(caracteristicas[0])
# print(caracteristicas[0].shape)

rotulo = np.load("Conjunto de Dados/caracteres-completo/Y_classe.npy")

estatisticas = np.zeros((100, 8))
# print(rotulo[0])

melhor_acerto = 0
melhor_alfa = 0
melhor_camada_escondida = 0

for alfa in range(1, 10):
    for camada_escondida in range(10, 110, 10):
        mlp = MLP(120, camada_escondida, 26, alfa/10);

        acertos = 0
        acertos_treinamento = 0

        erros = 0
        erros_treinamento = 0

        # print(caracteristicasDuasDimensoes.shape)
        # print(caracteristicas.shape[0])

        i = 1
        while ((not (np.array_equal(mlp.w, mlp.w_anterior) and np.array_equal(mlp.v, mlp.v_anterior))) and i < 1000): # Enquanto os pesos não convergirem
            mlp.v_anterior = mlp.v
            mlp.w_anterior = mlp.w

            for j in range(len(caracteristicas)-130):
                mlp.feedforward(caracteristicas[j])
                mlp.backpropagation(caracteristicas[j], rotulo[j])

            # print("Iteracao:", i, "Pesos v:", mlp.v, "Pesos w:", mlp.w, "Erro total:", mlp.erro_total)
            i += 1

        print("a")

        for j in range(len(caracteristicas)-130):
            resultado = mlp.feedforward(caracteristicas[j])
            if np.argmax(resultado) == np.argmax(rotulo[j]):
                acertos += 1
            else:
                erros += 1
        for j in range(len(caracteristicas)-130, len(caracteristicas)):
            resultado = mlp.feedforward(caracteristicas[j])
            resultado = np.heaviside(resultado - 0.5, 0)
            if np.argmax(resultado) == np.argmax(rotulo[j]):
                acertos += 1
                acertos_treinamento += 1
            else:
                erros += 1
                erros_treinamento += 1

        print("Alfa:", alfa/10, "Camada escondida:", camada_escondida, "Acertos:", acertos, "Erros:", erros, "Acuracia:", acertos/(acertos+erros), "Acertos treinamento:", acertos_treinamento, "Erros treinamento:", erros_treinamento, "Acuracia treinamento:", acertos_treinamento/(acertos_treinamento+erros_treinamento))
        estatisticas[(alfa-1)*10 + (camada_escondida//10)-1] = np.array([alfa/10, camada_escondida, acertos, erros, acertos/(acertos+erros), acertos_treinamento, erros_treinamento, acertos_treinamento/(acertos_treinamento+erros_treinamento)])

        acuracia = acertos_treinamento / (acertos_treinamento + erros_treinamento)
        if acuracia > melhor_acerto:
            melhor_acerto = acuracia
            melhor_alfa = alfa
            melhor_camada_escondida = camada_escondida

np.save("Testes/estatisticas/pesosZeradosTanh.npy", estatisticas) # salva as estatisticas em um arquivo numpy para poder analisar depois

print("MLP com função de ativação tanh e pesos zerados")
print("Melhor acerto:", melhor_acerto)
print("Melhor alfa:", melhor_alfa)
print("Melhor camada escondida:", melhor_camada_escondida)
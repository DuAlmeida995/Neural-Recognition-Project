import numpy as np

# funcao de ativacao sigmoide e sua derivada
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

class MLP:
    def __init__(self, n_entrada, n_escondida, n_saida, taxa_aprendizado):
        self.n_entrada = n_entrada
        self.n_escondida = n_escondida
        self.n_saida = n_saida
        self.alpha = taxa_aprendizado

        # peso entre camada de entrada e camada escondida (matriz v)
        self.v = np.random.uniform(-1, 1, (self.n_escondida, self.n_entrada + 1)) # o +1 em v e w se deve por conta do bias
        # peso entre camada escondida e camada de saída (matriz w)
        self.w = np.random.uniform(-1, 1, (self.n_saida, self.n_escondida + 1))   

    def feedforward(self, x):
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
        x = np.insert(x, 0, 1) 

        # calculo do erro na camada de saida
        erro_saida = t - self.y
        self.delta_saida = erro_saida * derivada_sigmoide(self.y_in)

        # retropropagacao do erro para a camada escondida
        w_sem_bias = self.w[:, 1:] # remove o bias da matriz w para calcular o erro escondido
        erro_escondida = np.dot(self.delta_saida, w_sem_bias)
        self.delta_escondida = erro_escondida * derivada_sigmoide(self.z)
        delta_w = self.alpha * np.outer(self.delta_saida, self.a) # utilizei o np.outer aqui por conta que ele consegue pegar dois vetores e criar uma matriz onde cada posição (i, j) é o resultado de a[i] X b[j]
        delta_v = self.alpha * np.outer(self.delta_escondida, x)
        self.w = self.w + delta_w
        self.v = self.v + delta_v 

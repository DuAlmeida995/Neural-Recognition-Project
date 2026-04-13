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

        # calculo da camada de saída
        self.y_in = np.dot(self.w, self.a)
        self.y = sigmoide (self.y_in)
        return self.y
    


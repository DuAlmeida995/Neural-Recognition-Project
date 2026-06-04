# =============================================================================
# Criação de um Arquivo Modificado do Conjunto de Dados de Teste da MLP
# =============================================================================
# Este script carrega o conjunto de dados e motifica os últimos 130 elementos,
# que são usados como conjunto de teste na MLP, através de pequenas alterações
# aleatórias nos caracteres, cortando uma linha da letra ou alterando 6 pixels
# =============================================================================

import numpy as np
import random
import matplotlib.pyplot as plt

# Carrega os dados originais
# X.npy -> shape (N, H, W): N imagens de altura H e largura W
X = np.load("Conjunto de Dados/caracteres-completo/X.npy")

# É feita uma cópia para não sobrescrever os dados
X_autoral = X.copy() 

# Separa o valor que é usado para o fundo e o valor que é usado para o traço da letra na imagem
fundo = np.max(X)
traco = np.min(X)

# -----------------------------------------------------------------------------
# Alteração do Conjunto de Dados para Testes
# -----------------------------------------------------------------------------

# Início da alteração dos dados do conjunto de teste
print("Alterando o conjunto de teste...")

# Define o início dos dados de teste (os últimos 130 elementos)
inicio_teste = len(X) - 130

#------------------------------------------------------
# Escolhe qual letra será visualizada
#indice = inicio_teste
#imagem_letra = X_autoral[indice]

# Configuração da exibição da imagem
# O cmap='gray' -> preto e branco
#plt.imshow(imagem_letra, cmap='gray')
#plt.title(f"Visualização da letra no índice {indice}")
#plt.show()
#------------------------------------------------------

for i in range(inicio_teste, len(X)):
    # Pega a letra atual (12x10)
    letra = X_autoral[i] 
    
    # Sorteia qual defeito aplicar nessa letra
    # - corte: realiza um corte em uma linha aleatória da letra
    # - chuvisco: inverte 12 pixels aleatórios da imagem
    defeito = random.choice(["corte", "chuvisco"])
    
    if defeito == "corte":
        linha = random.randint(0, 9)   # Sorteia uma linha de 0 a 11
        letra[ linha, :] = fundo
        
    elif defeito == "chuvisco":
        # Sorteia 12 pixels aleatórios na imagem e inverte a cor deles
        for j in range(6):
            linha = random.randint(0, 9)   # Sorteia uma linha de 0 a 9
            coluna = random.randint(0, 11)   # Sorteia uma coluna de 0 a 12
            
            # Se o pixel pertencer ao traço da letra, ele é apagado
            # Se o pixel pertencer ao fundo da letra, ele é preenchido
            if letra[linha, coluna] == traco:
                letra[linha, coluna] = fundo
            else:
                letra[linha, coluna] = traco

# Salva o novo arquivo com o conjunto de dados de teste alterado
np.save("Conjunto de Dados/caracteres-completo/X_autoral.npy", X_autoral)
print("Arquivo 'X_autoral.npy' gerado")

#------------------------------------------------------
# Escolhe qual letra será visualizada
#indice = inicio_teste
#imagem_letra = X_autoral[indice]

# Configuração da exibição da imagem
# O cmap='gray' -> preto e branco
#plt.imshow(imagem_letra, cmap='gray')
#plt.title(f"Visualização da letra no índice {indice}")
#plt.show()
#------------------------------------------------------


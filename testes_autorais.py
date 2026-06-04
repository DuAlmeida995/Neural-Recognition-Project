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

# carrega e copia os dados originais
X = np.load("Conjunto de Dados/caracteres-completo/X.npy")
X_autoral = X.copy()  # copia pra não sobrescrever o original

fundo = np.max(X)  # valor do background
traco = np.min(X)  # valor do traço da letra

print("Alterando o conjunto de teste...")
inicio_teste = len(X) - 130  # últimos 130 são o conjunto de teste

#------------------------------------------------------
# Escolhe qual letra será visualizada
# indice = inicio_teste
# imagem_letra = X_autoral[indice]

# Configuração da exibição da imagem
# O cmap='gray' -> preto e branco
# plt.imshow(imagem_letra, cmap='gray')
# plt.title(f"Visualização da letra no índice {indice}")
# plt.show()
#------------------------------------------------------

for i in range(inicio_teste, len(X)):
    letra = X_autoral[i]  # imagem 12x10
    
    # sorteia defeito: corte apaga uma linha inteira, chuvisco inverte 6 pixels
    defeito = random.choice(["corte", "chuvisco"])
    
    if defeito == "corte":
        linha = random.randint(0, 9)
        letra[ linha, :] = fundo
        
    elif defeito == "chuvisco":
        # inverte 6 pixels aleatórios
        for j in range(6):
            linha = random.randint(0, 9)   # Sorteia uma linha de 0 a 9
            coluna = random.randint(0, 11)   # Sorteia uma coluna de 0 a 11
            
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


# Neural-Recognition-Project

Este repositório contém o desenvolvimento e a implementação manual de uma rede neural artificial **Multilayer Perceptron (MLP)** para o reconhecimento e classificação de caracteres alfabéticos.

---

## Integrantes do Grupo
* **André Portela Lino** - 15634885
* **Davi Lima de Oliveira** - 15648741
* **Eduardo Almeida Cavalcanti de Melo** - 15526004
* **Eric Isin Wang Chou** - 15574579
* **Júlio Arroio Silva** - 15466241
* **Karina Yang Chen** - 15466658

---

## Sobre a Multilayer Perceptron (MLP)
Implementação matemática pura da arquitetura **Feedforward** com algoritmo de treinamento **Backpropagation** baseado em **Gradiente Descendente**.
### Principais Características Implementadas:
* **Camadas:** Uma única camada escondida configurável com pesos inicializados via distribuição uniforme estável.
* **Função de Perda:** Erro Quadrático Médio (MSE - *Mean Squared Error*).
* **Mecanismos de Parada:** Critério de convergência por iterações e **Parada Antecipada (Early Stopping)** automática baseada no monitoramento de estagnação de erro no conjunto de validação.
* **Validação (Hold-Out):** Divisão explícita das amostras do dataset em três conjuntos independentes e isolados em memória: **Treinamento** (para ajuste aditivo de pesos), **Validação** (para controle do Early Stopping) e **Teste** (reservado exclusivamente para avaliação final de acurácia).

---

## Estrutura de Arquivos do Projeto

* `mlp.py`: Código principal contendo a classe da rede neural, a rotina de treinamento e a avaliação final.
* `testes_autorais.py`: Script responsável por carregar o dataset original e criar uma versão modificada aplicando ruídos customizados (*corte de linha* e *chuvisco*) nas instâncias de teste.
* `visualizar_matriz.py`: Script utilitário que lê os dados de saída e gera a exibição gráfica da Matriz de Confusão em um mapa de calor.
* `visualizar_erros.py`: Utilitário de monitoramento que carrega o histórico bruto de perdas (`historico_erros.txt`) e gera automaticamente o gráfico visual da evolução do Erro Quadrático Médio (MSE).

### Relatórios de Saída (Pasta `/saidas`)
O algoritmo exporta automaticamente para a pasta `saidas/` os seguintes arquivos de texto legíveis exigidos para a entrega:
* `hiperparametros_teste_1.txt`: Configurações da arquitetura da rede (Camadas, Alphas, etc.).
* `pesos_iniciais_v_teste_1.txt` e `pesos_iniciais_w_teste_1.txt`: Estado inicial das matrizes de pesos antes do treino.
* `pesos_finais_v_teste_1.txt` e `pesos_finais_w_teste_1.txt`: Estado final das matrizes de pesos ajustados.
* `historico_erros_teste_1.txt`: Histórico detalhado do erro cometido pela rede em cada época do treinamento.
* `saidas_produzidas_teste_1.txt`: Saídas contínuas (predições) geradas pela rede para cada um dos dados de teste.
* `matriz_confusao_teste_1.txt`: Matriz plano de confusão $26 \times 26$ contendo o cruzamento de predições.

---

## Como Executar o Projeto

### 1. Preparar o Ambiente Virtual (Recomendado)

Para evitar conflitos com pacotes globais do Python, crie e ative o ambiente virtual na pasta do projeto:
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar no Windows (Prompt de Comando)
.venv\Scripts\activate
# Ou ativar no Linux/Mac/Git Bash
source .venv/bin/activate
```
### 2. Instalar as Dependências Permissíveis de I/O e Plot

```bash
pip install numpy matplotlib seaborn
```

### 3. Executar o Fluxo Completo do Trabalho

**Passo A: (Opcional) Gerar os dados com variações autorais**
Para estressar a rede com o conjunto ruidoso modificado pelo grupo, execute:
```bash
python testes_autorais.py
```

**Passo B: Executar o Treinamento e Teste da MLP**
Rode o script principal para treinar a rede no dataset de caracteres completo e exportar os relatórios:
```bash
python mlp.py
```
**Passo C: Plotar a Curva de Comportamento de Erros**
Para processar os logs de perda e renderizar o gráfico do MSE por época, execute:
```bash
python visualizar_erros.py
```

**Passo D: Plotar o Gráfico da Matriz de Confusão**
Após a finalização do teste, gere a visualização gráfica (mapa de calor):
```bash 
python visualizar_matriz.py
```



import random
import math
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parâmetros do problema
A = 10
p = 20
dimensoes = p
limite_inferior = -10
limite_superior = 10
tamanho_populacao = 100
taxa_cruzamento = 0.7
taxa_mutacao = 0.01
num_geracoes = 100
bits_per_gene = 10

def rastrigin(x):
    return A * p + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

def inicializar_populacao():
    return [np.random.uniform(limite_inferior, limite_superior, dimensoes) for _ in range(tamanho_populacao)]

def calcular_aptidao(individuo):
    return rastrigin(individuo) + 1

def selecao_roleta(populacao):
    adaptacoes = [calcular_aptidao(individuo) for individuo in populacao]
    total_adaptacoes = sum(adaptacoes)
    probabilidades_selecao = [adaptacao / total_adaptacoes for adaptacao in adaptacoes]
    return populacao[np.random.choice(len(populacao), p=probabilidades_selecao)]

def cruzamento(pai1, pai2):
    if random.random() < taxa_cruzamento:
        ponto = random.randint(1, dimensoes - 1)
        filho1 = np.concatenate((pai1[:ponto], pai2[ponto:]))
        filho2 = np.concatenate((pai2[:ponto], pai1[ponto:]))
        return filho1, filho2
    else:
        return pai1, pai2

def mutacao(individuo):
    for i in range(dimensoes):
        if random.random() < taxa_mutacao:
            individuo[i] = np.random.uniform(limite_inferior, limite_superior)

def algoritmo_genetico():
    populacao = inicializar_populacao()
    aptidoes = []
    for _ in range(num_geracoes):
        nova_populacao = []
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_roleta(populacao)
            pai2 = selecao_roleta(populacao)
            filho1, filho2 = cruzamento(pai1, pai2)
            mutacao(filho1)
            mutacao(filho2)
            nova_populacao.extend([filho1, filho2])
        populacao = nova_populacao
        aptidoes.append(min(calcular_aptidao(individuo) for individuo in populacao))
    return aptidoes

# Execução do algoritmo genético
aptidoes = algoritmo_genetico()

# Geração do gráfico
plt.plot(aptidoes)
plt.title("Evolução da Aptidão")
plt.xlabel("Gerações")
plt.ylabel("Aptidão")
plt.grid(True)
plt.show()


resultados = {
    'Estatística': ['Menor Valor de Aptidão', 'Maior Valor de Aptidão', 'Média de Aptidão', 'Desvio-Padrão de Aptidão'],
    'Valor': [min(aptidoes), max(aptidoes), np.mean(aptidoes), np.std(aptidoes)]
}
# Criando um DataFrame com os resultados
df_resultados = pd.DataFrame(resultados)

# Exibindo a tabela
print(df_resultados)
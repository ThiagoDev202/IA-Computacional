import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

qnt_individuos = 50
qnt_max_geracoes = 100
valor_otimo_aceitavel = 10
Numero_elites = 5
numero_execucoes = 100

def ler_pontos_arquivo(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, header=None)
    return df.values

def inicializar_populacao(tamanho, numero_pontos):
    return [np.random.permutation(numero_pontos) for _ in range(tamanho)]

def funcao_aptidao(cromossomo, pontos):
    soma_distancias = 0
    for i in range(len(cromossomo)):
        ponto_atual = pontos[cromossomo[i]]
        ponto_proximo = pontos[cromossomo[(i + 1) % len(cromossomo)]]
        distancia = np.linalg.norm(ponto_atual - ponto_proximo)
        soma_distancias += distancia
    return soma_distancias

def torneio(populacao, aptidoes, tamanho_torneio=3):
    selecionados = []
    while len(selecionados) < len(populacao):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        melhor = competidores[np.argmin(aptidoes[competidores])]
        selecionados.append(populacao[melhor])
    return selecionados

def crossover_dois_pontos(pai1, pai2):
    tamanho = len(pai1)
    filho1, filho2 = pai1.copy(), pai2.copy()
    ponto1, ponto2 = sorted(np.random.choice(range(1, tamanho-1), 2, replace=False))
    filho1[ponto1:ponto2], filho2[ponto1:ponto2] = pai2[ponto1:ponto2], pai1[ponto1:ponto2]
    return filho1, filho2

def mutacao_por_troca(cromossomo, taxa_de_mutacao=0.05):
    cromossomo = np.array(cromossomo)
    for i in range(len(cromossomo)):
        if np.random.rand() < taxa_de_mutacao:
            j = np.random.randint(len(cromossomo))
            cromossomo[i], cromossomo[j] = cromossomo[j], cromossomo[i]
    return cromossomo.tolist()

def elitismo(populacao, aptidoes, n_elites):
    elite_indices = np.argsort(aptidoes)[:n_elites]
    return [populacao[i] for i in elite_indices]

def plot_tsp_3d(pontos, cromossomo):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = pontos[:, 0], pontos[:, 1], pontos[:, 2]
    ax.scatter(x, y, z, color='blue', s=100, zorder=5)
    ax.scatter(x[0], y[0], z[0], color='red', s=200, zorder=5, label='Origem')
    for i in range(len(cromossomo) - 1):
        start_pos = pontos[cromossomo[i]]
        end_pos = pontos[cromossomo[i+1]]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 'k-', zorder=1)
    ax.plot([pontos[cromossomo[-1]][0], pontos[0][0]], [pontos[cromossomo[-1]][1], pontos[0][1]], [pontos[cromossomo[-1]][2], pontos[0][2]], 'k-', zorder=1)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Coordenada Z')
    ax.legend()
    ax.grid(True)
    plt.show()

def algoritmo_genetico(pontos):
    populacao = inicializar_populacao(qnt_individuos, len(pontos))
    todas_aptidoes = []
    melhor_solucao = np.inf
    melhor_cromossomo = None

    for geracao in range(qnt_max_geracoes):
        aptidoes = np.array([funcao_aptidao(ind, pontos) for ind in populacao])
        todas_aptidoes.extend(aptidoes)
        if np.min(aptidoes) < melhor_solucao:
            melhor_solucao = np.min(aptidoes)
            melhor_cromossomo = populacao[np.argmin(aptidoes)]
            print(f"Geracao {geracao}: Melhor aptidao = {melhor_solucao}")

        if melhor_solucao <= valor_otimo_aceitavel:
            print("Condição de parada atingida.")
            break

        elites = elitismo(populacao, aptidoes, Numero_elites)
        selected = torneio(populacao, aptidoes)
        offspring = [crossover_dois_pontos(selected[i], selected[(i + 1) % len(selected)]) for i in range(0, len(selected), 2)]
        populacao = [mutacao_por_troca(child) for pair in offspring for child in pair]
        populacao.extend(elites)

    # Plotar o gráfico ao final das iterações, mostrando a melhor rota encontrada
    if melhor_cromossomo is not None:
        plot_tsp_3d(pontos, melhor_cromossomo)
    else:
        print("Nenhuma solução válida encontrada.")

    # Estatísticas
    min_aptidao = np.min(todas_aptidoes)
    max_aptidao = np.max(todas_aptidoes)
    media_aptidao = np.mean(todas_aptidoes)
    desvio_aptidao = np.std(todas_aptidoes)

    print("Menor Valor de Aptidão:", min_aptidao)
    print("Maior Valor de Aptidão:", max_aptidao)
    print("Média de Valor de Aptidão:", media_aptidao)
    print("Desvio-Padrão de Valor de Aptidão:", desvio_aptidao)

    return melhor_solucao

# Caminho para o arquivo CSV
caminho_arquivo = 'C:\\Users\\Windows\\Downloads\\CaixeiroSimples.csv'
pontos = ler_pontos_arquivo(caminho_arquivo)

# Executa o algoritmo genético usando os pontos lidos do arquivo
melhor_aptidao = algoritmo_genetico(pontos)
print("Melhor aptidão encontrada:", melhor_aptidao)

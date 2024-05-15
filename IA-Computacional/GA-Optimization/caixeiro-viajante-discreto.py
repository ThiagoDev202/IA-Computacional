import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuração inicial
qnt_individuos = 50
qnt_max_geracoes = 100
valor_otimo_aceitavel = 10
Numero_elites = 5

# Leitura dos dados
def ler_pontos_arquivo(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, header=None)
    pontos = df.iloc[:, :3].values
    grupos = df.iloc[:, 3].astype(int).values
    return pontos, grupos

# Inicialização da população
def inicializar_populacao(tamanho, numero_pontos):
    return [np.random.permutation(numero_pontos) for _ in range(tamanho)]

# Função de aptidão
def funcao_aptidao(cromossomo, pontos, grupos):
    soma_distancias = 0
    ultimo_grupo = grupos[cromossomo[0]]
    for i in range(1, len(cromossomo)):
        if grupos[cromossomo[i]] == ultimo_grupo:
            ponto_atual = pontos[cromossomo[i-1]]
            ponto_proximo = pontos[cromossomo[i]]
            soma_distancias += np.linalg.norm(ponto_atual - ponto_proximo)
        else:
            # Penalização por mudar de grupo prematuramente
            soma_distancias += 10000  # Ajuste o valor conforme a severidade desejada
    return soma_distancias

# Seleção por torneio
def torneio(populacao, aptidoes, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        melhor = competidores[np.argmin([aptidoes[i] for i in competidores])]
        selecionados.append(populacao[melhor])
    return selecionados

# Crossover de dois pontos
def crossover_dois_pontos(pai1, pai2):
    tamanho = len(pai1)
    filho1, filho2 = pai1.copy(), pai2.copy()
    ponto1, ponto2 = sorted(np.random.choice(range(1, tamanho-1), 2, replace=False))
    filho1[ponto1:ponto2], filho2[ponto1:ponto2] = pai2[ponto1:ponto2], pai1[ponto1:ponto2]
    return filho1, filho2

# Mutação por troca
def mutacao_por_troca(cromossomo, taxa_de_mutacao=0.05):
    for _ in range(int(taxa_de_mutacao * len(cromossomo))):
        i, j = np.random.randint(0, len(cromossomo), 2)
        cromossomo[i], cromossomo[j] = cromossomo[j], cromossomo[i]
    return cromossomo

# Elitismo
def elitismo(populacao, aptidoes, n_elites):
    elite_indices = np.argsort(aptidoes)[:n_elites]
    return [populacao[i] for i in elite_indices]

# Plot do TSP 3D
def plot_tsp_3d(pontos, cromossomo, grupos):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    grupo_cores = ['red', 'green', 'blue', 'purple', 'orange', 'yellow']
    for g in np.unique(grupos):
        ix = grupos == g
        ax.scatter(pontos[ix, 0], pontos[ix, 1], pontos[ix, 2], color=grupo_cores[g], label=f'Grupo {g}', s=100)
    for i in range(len(cromossomo) - 1):
        start_pos = pontos[cromossomo[i]]
        end_pos = pontos[cromossomo[i+1]]
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], [start_pos[2], end_pos[2]], 'k-', zorder=1)
    ax.set_xlabel('Coordenada X')
    ax.set_ylabel('Coordenada Y')
    ax.set_zlabel('Coordenada Z')
    ax.legend()
    plt.show()

# Algoritmo Genético
def algoritmo_genetico(pontos, grupos):
    populacao = inicializar_populacao(qnt_individuos, len(pontos))
    melhor_solucao = np.inf
    melhor_cromossomo = None
    aptidoes_hist = []
    aptidoes_gerais = []

    for geracao in range(qnt_max_geracoes):
        aptidoes = np.array([funcao_aptidao(ind, pontos, grupos) for ind in populacao])
        melhor_aptidao = np.min(aptidoes)
        aptidoes_hist.append(melhor_aptidao)
        aptidoes_gerais.extend(aptidoes)
        if melhor_aptidao < melhor_solucao:
            melhor_solucao = melhor_aptidao
            melhor_cromossomo = populacao[np.argmin(aptidoes)]
            print(f'Geracao {geracao}: Melhor aptidao = {melhor_aptidao}')
        if melhor_aptidao <= valor_otimo_aceitavel:
            print("Condição de parada atingida.")
            break
        selecionados = torneio(populacao, aptidoes)
        novos_individuos = []
        for i in range(0, len(selecionados), 2):
            if i + 1 < len(selecionados):
                filho1, filho2 = crossover_dois_pontos(selecionados[i], selecionados[i + 1])
                novos_individuos.append(mutacao_por_troca(filho1))
                novos_individuos.append(mutacao_por_troca(filho2))
        populacao = elitismo(populacao, aptidoes, Numero_elites) + novos_individuos

    if melhor_cromossomo is not None:
        plot_tsp_3d(pontos, melhor_cromossomo, grupos)
    else:
        print("Nenhuma solução válida encontrada.")

    # Estatísticas finais
    print("Menor Valor de Aptidão:", np.min(aptidoes_hist))
    print("Maior Valor de Aptidão:", np.max(aptidoes_hist))
    print("Média de Valor de Aptidão:", np.mean(aptidoes_hist))
    print("Desvio-Padrão de Valor de Aptidão:", np.std(aptidoes_hist))

    return melhor_solucao

# Inicialização e execução
caminho_arquivo = 'C:\\IA-Computacional\\GA-Optimization\\CaixeroGrupos.csv'
pontos, grupos = ler_pontos_arquivo(caminho_arquivo)
melhor_aptidao = algoritmo_genetico(pontos, grupos)
print("Melhor aptidão encontrada:", melhor_aptidao)

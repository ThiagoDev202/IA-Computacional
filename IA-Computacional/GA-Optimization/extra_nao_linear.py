import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Carregar os dados (substitua 'caminho/para/aerogerador.dat' pelo caminho real do arquivo)
caminho_arquivo = 'aerogerador.dat'
dados = pd.read_csv(caminho_arquivo, sep='\t', header=None, names=['Velocidade', 'Potencia'])

# Plotar o gráfico de dispersão
plt.figure(figsize=(8, 6))
plt.scatter(dados['Velocidade'], dados['Potencia'], color='b', alpha=0.5)
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Relação entre Velocidade do Vento e Potência Gerada')
plt.grid(True)
plt.show()

# Definir o número de rodadas
num_rodadas = 1000

# Inicializar listas para armazenar resultados
eqm_mqo_tradicional = []
eqm_mqo_regularizado = []
eqm_media_valores = []

for _ in range(num_rodadas):
    # Embaralhar os dados e dividir em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(dados['Velocidade'], dados['Potencia'], test_size=0.2, shuffle=True)

    # Modelo MQO Tradicional
    modelo_mqo = LinearRegression()
    modelo_mqo.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred_mqo = modelo_mqo.predict(X_test.values.reshape(-1, 1))
    eqm_mqo_tradicional.append(mean_squared_error(y_test, y_pred_mqo))

    # Modelo MQO Regularizado (Ridge)
    modelo_ridge = Ridge(alpha=1.0)  # Ajuste o valor de alpha conforme necessário
    modelo_ridge.fit(X_train.values.reshape(-1, 1), y_train)
    y_pred_ridge = modelo_ridge.predict(X_test.values.reshape(-1, 1))
    eqm_mqo_regularizado.append(mean_squared_error(y_test, y_pred_ridge))

    # Média dos Valores Observados
    media_valores = y_train.mean()
    eqm_media_valores.append(mean_squared_error(y_test, [media_valores] * len(y_test)))

# Calcular estatísticas dos EQMs
media_mqo_tradicional = sum(eqm_mqo_tradicional) / num_rodadas
media_mqo_regularizado = sum(eqm_mqo_regularizado) / num_rodadas
media_media_valores = sum(eqm_media_valores) / num_rodadas

# Exibir resultados
print(f"EQM MQO Tradicional: {media_mqo_tradicional:.2f}")
print(f"EQM MQO Regularizado: {media_mqo_regularizado:.2f}")
print(f"EQM Média dos Valores: {media_media_valores:.2f}")

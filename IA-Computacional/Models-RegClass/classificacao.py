import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Carregar o arquivo EMGDataset.csv
df = pd.read_csv('C:\\IA-Computacional\\Models-RegClass\\EMGDataset.csv')

# Organizar os dados nas matrizes X e Y
X = df.iloc[:, :2].values  # Características (Sensor 1 e Sensor 2)
Y = df.iloc[:, 2].values   # Rótulos das classes

# Visualização inicial dos dados (gráfico de dispersão)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', alpha=0.5)
plt.colorbar()
plt.xlabel('Sensor 1')
plt.ylabel('Sensor 2')
plt.title('Gráfico de Dispersão dos Dados')
plt.show()

# Definir quantidade de rodadas
n_rodadas = 100

# Inicializar listas para armazenar acurácias
acuracia_mqo = []
acuracia_tikhonov = []

# Hiperparâmetro para Tikhonov (regularização Ridge)
alpha_values = np.logspace(-4, 4, 10)

for alpha in alpha_values:
    acuracia_temp = []
    for _ in range(n_rodadas):
        # Embaralhar os dados e particionar
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

        # Treinar modelo MQO regularizado (Tikhonov)
        tikhonov_model = Ridge(alpha=alpha)
        tikhonov_model.fit(X_train, Y_train)

        # Prever rótulos
        Y_pred_tikhonov = tikhonov_model.predict(X_test)

        # Calcular acurácia
        acuracia_temp.append(accuracy_score(Y_test, np.round(Y_pred_tikhonov)))
    
    acuracia_tikhonov.append(np.mean(acuracia_temp))
    print(f"Alpha: {alpha:.4f} - Acurácia Média: {np.mean(acuracia_temp):.4f}")

# Encontrar melhor alpha e calcular estatísticas para o melhor modelo
best_alpha = alpha_values[np.argmax(acuracia_tikhonov)]
print(f"Melhor Alpha: {best_alpha}")

# Estatísticas dos resultados
media = np.mean(acuracia_tikhonov)
desvio_padrao = np.std(acuracia_tikhonov)
maximo = np.max(acuracia_tikhonov)
minimo = np.min(acuracia_tikhonov)

print(f"MQO Regularizado (Tikhonov) com melhor Alpha: Média = {media:.4f}, Desvio Padrão = {desvio_padrao:.4f}, "
      f"Máximo = {maximo:.4f}, Mínimo = {minimo:.4f}")

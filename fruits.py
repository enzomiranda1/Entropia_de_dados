import pandas as pd
import numpy as np
from scipy.stats import entropy

# Definir as frutas disponíveis
fruits = ['Apple', 'Banana', 'Orange', 'Mango']

# Gerar o dataset aleatório
np.random.seed(42)  # Para reprodutibilidade
choices = np.random.choice(fruits, size=40)

# Criar o DataFrame com as escolhas das pessoas
data = pd.DataFrame({'fruits': choices})

# Calcular a frequência de ocorrência de cada fruta
class_counts = data['fruits'].value_counts()

# Calcular a probabilidade de cada fruta
class_probabilities = class_counts / len(data)

# Calcular a entropia do conjunto de dados
class_entropy = entropy(class_probabilities)

# Calcular a entropia máxima do conjunto de dados
max_entropy = entropy([1 / len(fruits)] * len(fruits))

print("Entropia do dataset: {:.4f}".format(class_entropy))
print("Entropia Máxima do dataset: {:.4f}".format(max_entropy))

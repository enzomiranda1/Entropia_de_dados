import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from math import log2

# Carregar o dataset "Mushroom" do UCI Machine Learning Repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
column_names = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing',
                'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring',
                'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
dataset = pd.read_csv(url, names=column_names)

# Remover colunas desnecessárias
dataset.drop(['stalk_root'], axis=1, inplace=True)

# Converter atributos qualitativos para numéricos usando one-hot encoding
qualitativos_encoded = pd.get_dummies(dataset.drop('class', axis=1))

# Definir atributos quantitativos e a coluna alvo
quantitativos = pd.DataFrame()  # Não há atributos quantitativos neste dataset
alvo = dataset['class'].map({'p': 0, 'e': 1})  # Convertendo a classe para valores numéricos (p = 0, e = 1)

# Definir o número de clusters (grupos)
k = len(dataset['class'].unique())

# Executar o algoritmo k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(qualitativos_encoded, alvo)

# Fazer a predição dos rótulos dos grupos
rotulos = knn.predict(qualitativos_encoded)

# Calcular a entropia dos grupos
def calcular_entropia(grupos):
    n_total = sum(len(g) for g in grupos)
    entropia = 0

    for grupo in grupos:
        n_grupo = len(grupo)
        proporcao = n_grupo / n_total

        if proporcao > 0:
            entropia_grupo = proporcao * log2(proporcao)
            entropia -= entropia_grupo

    return entropia

grupos = [alvo[rotulos == i] for i in range(k)]
entropia = calcular_entropia(grupos)

# Calcular a entropia máxima
entropia_maxima = log2(k)

print("Entropia:", entropia)
print("Entropia Máxima:", entropia_maxima)

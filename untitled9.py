def calculate_entropy(y):
    # Função para calcular a entropia de uma variável categórica
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return entropy(probabilities, base=2)

def calculate_max_entropy(y):
    # Função para calcular a entropia máxima de uma variável categórica
    n_classes = len(np.unique(y))
    return np.log2(n_classes)
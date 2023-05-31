import seaborn as sns
import numpy as np
from scipy.stats import entropy

# Carregar o conjunto de dados Iris do Seaborn
tips = sns.load_dataset('iris')

# Calcular a entropia e entropia máxima para cada atributo
for column in tips.columns:
    target = tips[column]
    classes, counts = np.unique(target, return_counts=True)
    total_count = len(target)
    num_classes = len(classes)
    
    class_probabilities = counts / total_count
    class_entropy = entropy(class_probabilities)
    max_entropy = np.log2(num_classes)
    
    print(f"Atributo {column}: Entropia = {class_entropy:.4f}, Entropia Máxima = {max_entropy:.4f}")

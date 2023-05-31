import pandas as pd
import numpy as np
from scipy.stats import entropy

# Criar o conjunto de dados "Play Tennis"
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'Play Tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

df = pd.DataFrame(data)

# Calcular a entropia e entropia máxima para cada atributo
for column in df.columns[:-1]:
    target = df[column]
    classes, counts = np.unique(target, return_counts=True)
    total_count = len(target)
    num_classes = len(classes)
    
    class_probabilities = counts / total_count
    class_entropy = entropy(class_probabilities)
    max_entropy = np.log2(num_classes)
    
    print(f"Atributo {column}: Entropia = {class_entropy:.4f}, Entropia Máxima = {max_entropy:.4f}")

import seaborn as sns
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer

# Carregar o conjunto de dados Iris do Seaborn
tips = sns.load_dataset('iris')

# Converter as colunas numéricas em qualitativas usando a discretização
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
tips_quantitativos = tips.select_dtypes(include=[np.number])
tips_qualitativos = discretizer.fit_transform(tips_quantitativos)
tips_discretizados = tips.copy()
tips_discretizados[tips_quantitativos.columns] = tips_qualitativos

# Mapear as classes para valores numéricos
class_mapping = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
tips_discretizados['species'] = tips_discretizados['species'].map(class_mapping)

# Calcular a entropia e entropia máxima para cada atributo
for column in tips_discretizados.columns:
    target = tips_discretizados[column]
    classes, counts = np.unique(target, return_counts=True)
    total_count = len(target)
    num_classes = len(classes)
    
    class_probabilities = counts / total_count
    class_entropy = entropy(class_probabilities, base=2)
    max_entropy = np.log2(num_classes)
    
    print(f"Atributo {column}: Entropia = {class_entropy:.4f}, Entropia Máxima = {max_entropy:.4f}")

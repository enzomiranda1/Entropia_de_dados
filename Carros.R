library(entropy)

# Carregar o conjunto de dados "Car Evaluation"
dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header = FALSE)

# Definir os nomes das colunas
colnames(dataset) <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")

# Selecionar as colunas com as classes qualitativas
qualitative_columns <- c("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")

# Calcular a entropia normal e máxima para cada classe
for (column in qualitative_columns) {
  class_entropy <- entropy(dataset[[column]])
  
  normal_entropy <- sum(class_entropy$prob * log2(class_entropy$prob))
  normal_entropy <- -normal_entropy
  
  max_entropy <- log2(length(class_entropy$class))
  
  print(paste("Coluna:", column))
  print(paste("Entropia Normal:", normal_entropy))
  print(paste("Entropia Máxima:", max_entropy))
  print("-----------------------------------------")
}

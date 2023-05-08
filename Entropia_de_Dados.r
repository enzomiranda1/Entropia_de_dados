# definindo as funções de entropia
entropy <- function(x) {
  probs <- table(x)/length(x)
  -sum(probs * log2(probs))
}

max_entropy <- function(n) {
  log2(n)
}

# carregando o pacote e conjunto de dados
library(datasets)
data("ChickWeight")

# calculando a entropia de dados para a variável "Diet"
diet_entropy <- entropy(ChickWeight$Diet)
cat("Entropia de dados para a variável 'Diet':", diet_entropy, "\n")

# calculando a entropia máxima para a variável "Diet"
diet_max_entropy <- max_entropy(length(unique(ChickWeight$Diet)))
cat("Entropia máxima para a variável 'Diet':", diet_max_entropy, "\n")

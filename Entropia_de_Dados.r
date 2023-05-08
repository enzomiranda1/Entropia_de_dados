entropy <- function(x) {
  probs <- table(x)/length(x)
  -sum(probs * log2(probs))
}

max_entropy <- function(n) {
  log2(n)
}

library(datasets)
data("ChickWeight")

diet_entropy <- entropy(ChickWeight$Diet)
cat("Entropia de dados para a variável 'Diet':", diet_entropy, "\n")

diet_max_entropy <- max_entropy(length(unique(ChickWeight$Diet)))
cat("Entropia máxima para a variável 'Diet':", diet_max_entropy, "\n")

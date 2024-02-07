library("scmintMR")
suppressMessages(library("tidyverse"))
library("ggplot2")
library("MendelianRandomization")

data(example_data)
names(example_data)
L <- length(example_data$gamma_hat)
K1 <- K2 <- 5
L
group <- list(exposure1 = 1:K1, exposure2 = 1:K2+K1)
set.seed(1)
# Example correlation matrix for sample overlap. In real data, it can be pre-estimated.
C <- 0.9 * diag(1,(1+K1+K2)) + 
  0.1 * matrix(1,nrow=1+K1+K2,ncol=1+K1+K2)

Lambda <- solve(C)

res <- mintMR(gammah = example_data$gamma_hat,
              Gammah = example_data$Gamma_hat,
              se1 = example_data$se_g,
              se2 = example_data$se_G,
              corr_mat = example_data$LD,
              reference = example_data$latent_status,
              Lambda = Lambda,
              PC1 = 6)

Q <- res$res$QRes %>% lapply(function(x) Reduce("+",x)/length(x))
W <- res$res$WRes %>% lapply(function(x) Reduce("+",x)/length(x))
P <- res$res$PRes %>% lapply(function(x) Reduce("+",x)/length(x))
SKATp <- mapply(function(q,w) SKAT::Get_Davies_PVal(q, w)$p.value, Q, W, SIMPLIFY = F) %>% 
  unlist()

hist(SKATp[rowSums(example_data$latent_status)==0])
hist(SKATp[rowSums(example_data$latent_status)>=1],breaks=20)
mean(SKATp[rowSums(example_data$latent_status)>=1]<0.05)

p <- data.frame(mintMR = as.vector(res$Pvalue))
p %>% 
  arrange(mintMR) %>% 
  mutate(unif = ppoints(n())) %>%
  mutate(value = -log10(mintMR), unif = -log10(unif)) %>%
  ggplot(aes(y = value, x = unif))+
  geom_abline(intercept = 0, slope = 1, alpha = 0.5)+
  geom_point(size = 3, col = "red", alpha=1)+
  theme_classic()+
  labs(y = expression(paste("Observed -log"[10], plain(P))),
       x = expression(paste("Expected -log"[10], plain(P))))

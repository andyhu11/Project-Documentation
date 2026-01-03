library(tidyverse)
library(PerformanceAnalytics)
library(moments)
library(quadprog)
library(gridExtra)

############################################
# Question 1
############################################

# (a)
data <- read.csv("daily_price_volume.csv")
data$Trddt <- as.Date(data$Trddt)


tickers <- c(678, 868, 600741, 600006, 600151)
adj_col <- "Adjprcnd"
df5 <- data %>%
  filter(Stkcd %in% tickers) %>%
  select(Stkcd, Trddt, Adjprcnd) %>%
  rename(AdjClose = Adjprcnd) %>%
  arrange(Stkcd, Trddt)

# compute log returns per stock
df_returns <- df5 %>%
  group_by(Stkcd) %>%
  arrange(Trddt) %>%
  mutate(LogRet = log(AdjClose) - log(lag(AdjClose))) %>%
  ungroup()

write.csv(df_returns, file="daily_price_volume_returns.csv",row.names = F)

# drop first NA for each stock
df_returns <- df_returns %>% filter(!is.na(LogRet))


# make wide matrix: rows = date, cols = tickers
ret_wide <- df_returns %>%
  select(Stkcd, Trddt, LogRet) %>%
  pivot_wider(names_from = Stkcd, values_from = LogRet, names_prefix = "S_") %>%
  arrange(Trddt)

ret_mat <- ret_wide %>% select(-Trddt) %>% drop_na()
dates_used <- ret_wide$Trddt[complete.cases(ret_wide %>% select(-Trddt))]

# convert to matrix and set column names as tickers
ret_matrix <- as.matrix(ret_mat)
colnames(ret_matrix) <- colnames(ret_mat)
nrow(ret_matrix)

# calculate the summary statistics
desc_stats <- function(x) {
  c(
    N = length(x),
    Mean = mean(x),
    Median = median(x),
    SD = sd(x),
    Skewness = skewness(x),
    Kurtosis = kurtosis(x),   # excess kurtosis by default in moments::kurtosis
    Min = min(x),
    Q1 = quantile(x, 0.25),
    Q3 = quantile(x, 0.75),
    Max = max(x)
  )
}

stats_list <- apply(ret_matrix, 2, desc_stats) %>% t() %>% as.data.frame()
round(stats_list, 6)



df_plot <- as.data.frame(ret_matrix) %>%
  mutate(Trddt = dates_used) %>%
  pivot_longer(cols = -Trddt, names_to = "Asset", values_to = "LogRet")

ggplot(df_plot, aes(x = Trddt, y = LogRet)) +
  geom_line() +
  facet_wrap(~ Asset, scales = "free_y", ncol = 1) +
  labs(title = "Daily Log Returns", x = "Date", y = "Log Return") +
  theme_minimal()


ggplot(df_plot, aes(x = LogRet)) +
  geom_histogram(aes(y = ..density..), bins = 80) +
  geom_density() +
  facet_wrap(~ Asset, scales = "free", ncol = 2) +
  labs(title = "Return Distributions") +
  theme_minimal()

# b)
set.seed(2025)
n_port <- 10000
n_assets <- ncol(ret_matrix)
mu <- colMeans(ret_matrix)        # expected daily returns
Sigma <- cov(ret_matrix)         # covariance (daily)
rf <- 0                          # set rf to zero for daily Sharpe

# compute portfolio stats
port_stats <- function(w, mu, Sigma, rf = 0) {
  port_ret <- sum(w * mu)
  port_sd  <- sqrt(as.numeric(t(w) %*% Sigma %*% w))
  sr <- ifelse(port_sd == 0, NA, (port_ret - rf) / port_sd)
  c(Return = port_ret, SD = port_sd, Sharpe = sr)
}

# generate unconstrained random weights 
randW_unconstrained <- matrix(rnorm(n_port * n_assets), nrow = n_port, ncol = n_assets)
randW_unconstrained <- t(apply(randW_unconstrained, 1, function(x) x / sum(x)))
colnames(randW_unconstrained) <- colnames(ret_matrix)

# compute stats for each random portfolio
ports_uncon <- t(apply(randW_unconstrained, 1, function(w) port_stats(w, mu, Sigma, rf)))
ports_uncon <- as.data.frame(ports_uncon)
ports_uncon$w1 <- randW_unconstrained[,1]
ports_uncon$w2 <- randW_unconstrained[,2]
ports_uncon$w3 <- randW_unconstrained[,3]
ports_uncon$w4 <- randW_unconstrained[,4]
ports_uncon$w5 <- randW_unconstrained[,5]


# find approximate GMV and Tangency among random portfolios
idx_gmv_rand <- which.min(ports_uncon$SD)
idx_tan_rand <- which.max(ports_uncon$Sharpe)

# random-sim GMV approx - Return, SD, Sharpe:
ports_uncon[idx_gmv_rand, c("Return","SD","Sharpe","w1","w2","w3","w4","w5")]
# random-sim Tangency approx - Return, SD, Sharpe
ports_uncon[idx_tan_rand, c("Return","SD","Sharpe","w1","w2","w3","w4","w5")]

# analytical solutions for unconstrained GMV and Tangency
one <- rep(1, n_assets)
invS <- solve(Sigma)

# unconstrained GMV weights
w_gmv_analytical <- invS %*% one / as.numeric(t(one) %*% invS %*% one)
w_gmv_analytical <- as.vector(w_gmv_analytical)
names(w_gmv_analytical) <- colnames(ret_matrix)

# unconstrained Tangency weights (Sharpe) with rf
w_tan_analytical_unnorm <- invS %*% (mu - rf)
w_tan_analytical <- as.vector(w_tan_analytical_unnorm / sum(w_tan_analytical_unnorm))
names(w_tan_analytical) <- colnames(ret_matrix)

# Analytical unconstrained GMV weights:
round(w_gmv_analytical, 6)
# Analytical unconstrained Tangency weights (normalized):
round(w_tan_analytical, 6)

# compute stats for analytical portfolios
stats_gmv <- port_stats(w_gmv_analytical, mu, Sigma, rf)
stats_tan <- port_stats(w_tan_analytical, mu, Sigma, rf)

# Analytical GMV stats (Return, SD, Sharpe
round(stats_gmv, 8)
# Analytical Tangency stats (Return, SD, Sharpe):
round(stats_tan, 8)

# efficient frontier plotting: use simulated points + overlay analytical GMV and Tangency
ef_df <- ports_uncon %>% select(Return, SD, Sharpe)
ef_df$type <- "Simulated"

ggplot(ef_df, aes(x = SD, y = Return)) +
  geom_point(alpha = 0.4, size = 0.8) +
  annotate("point", x = stats_gmv["SD"], y = stats_gmv["Return"], color = "red", size = 3) +
  annotate("point", x = stats_tan["SD"], y = stats_tan["Return"], color = "blue", size = 3) +
  annotate("text", x = stats_gmv["SD"], y = stats_gmv["Return"], label = "GMV", hjust = -0.1, vjust = 1.5, color = "red") +
  annotate("text", x = stats_tan["SD"], y = stats_tan["Return"], label = "Tangency", hjust = -0.1, vjust = 1.5, color = "blue") +
  labs(
    title = "Efficient Frontier (simulated portfolios)",
    x = "Portfolio Std Dev (daily)",
    y = "Portfolio Return (daily)"
  ) +
  theme_minimal()


# Constrained portfolios via simulation (no short, weight >=0, weight <= 0.25) 
# We will draw many random non-negative weight vectors and accept those meeting max weight <= 0.25
set.seed(2025)
n_try <- 200000     # number of draws; 
accepted <- matrix(NA, nrow = 0, ncol = n_assets)
accepted_stats <- data.frame()

batch <- 10000
i <- 0
while(nrow(accepted) < 10000 && i * batch < n_try) {
  i <- i + 1
  draws <- matrix(runif(batch * n_assets), ncol = n_assets)
  draws <- t(apply(draws, 1, function(x) x / sum(x)))
  # filter by max weight <= 0.25
  keep <- apply(draws, 1, function(x) all(x <= 0.25 + 1e-12))
  draws_kept <- draws[keep, , drop = FALSE]
  if(nrow(draws_kept) > 0) {
    accepted <- rbind(accepted, draws_kept)
  }
  # safety: break if too many iterations
  if(i >= 1000) break
}
if(nrow(accepted) == 0) stop("No constrained portfolios found. Increase n_try or relax bounds.")

# limit to first 10000 accepted if more
if(nrow(accepted) > 10000) accepted <- accepted[1:10000,]

colnames(accepted) <- tickers
con_ports_stats <- t(apply(accepted, 1, function(w) port_stats(w, mu, Sigma, rf)))
con_ports_stats <- as.data.frame(con_ports_stats)
con_ports_stats$w1 <- accepted[,1]
con_ports_stats$w2 <- accepted[,2]
con_ports_stats$w3 <- accepted[,3]
con_ports_stats$w4 <- accepted[,4]
con_ports_stats$w5 <- accepted[,5]

# pick constrained GMV (min SD) and constrained Tangency (max Sharpe)
idx_c_gmv <- which.min(con_ports_stats$SD)
idx_c_tan <- which.max(con_ports_stats$Sharpe)

# Constrained (no short, max 0.25) GMV stats and weights:
con_ports_stats[idx_c_gmv, c("Return","SD","Sharpe","w1","w2","w3","w4","w5")]

# Constrained (no short, max 0.25) Tangency stats and weights:
con_ports_stats[idx_c_tan, c("Return","SD","Sharpe","w1","w2","w3","w4","w5")]

# compare unconstrained analytical vs constrained (from accepted)
comparison <- list(
  Unconstrained_GMV = list(weights = w_gmv_analytical, stats = stats_gmv),
  Unconstrained_Tangency = list(weights = w_tan_analytical, stats = stats_tan),
  Constrained_GMV = list(weights = as.numeric(con_ports_stats[idx_c_gmv, c("w1","w2","w3","w4","w5")]),
                         stats = as.numeric(con_ports_stats[idx_c_gmv, c("Return","SD","Sharpe")])),
  Constrained_Tangency = list(weights = as.numeric(con_ports_stats[idx_c_tan, c("w1","w2","w3","w4","w5")]),
                              stats = as.numeric(con_ports_stats[idx_c_tan, c("Return","SD","Sharpe")]))
)

# Summary table: Returns (daily), SD (daily), Sharpe (daily)
tibble(
  Portfolio = c("Unc_GMV","Unc_Tan","Con_GMV","Con_Tan"),
  Return = c(stats_gmv["Return"], stats_tan["Return"], 
             con_ports_stats[idx_c_gmv,"Return"], con_ports_stats[idx_c_tan,"Return"]),
  SD = c(stats_gmv["SD"], stats_tan["SD"], 
         con_ports_stats[idx_c_gmv,"SD"], con_ports_stats[idx_c_tan,"SD"]),
  Sharpe = c(stats_gmv["Sharpe"], stats_tan["Sharpe"], 
             con_ports_stats[idx_c_gmv,"Sharpe"], con_ports_stats[idx_c_tan,"Sharpe"])
) %>% mutate(across(-Portfolio, ~ round(., 8)))

# print constrained weights nicely
# Constrained GMV weights (tickers):
round(as.numeric(con_ports_stats[idx_c_gmv, c("w1","w2","w3","w4","w5")]), 6)
names <- colnames(ret_matrix)
names
# Constrained Tangency weights (tickers):
round(as.numeric(con_ports_stats[idx_c_tan, c("w1","w2","w3","w4","w5")]), 6)

# save weights as lists 
weights_table <- tibble(
  Asset = names,
  Unc_GMV = round(w_gmv_analytical, 6),
  Unc_Tan = round(w_tan_analytical, 6),
  Con_GMV = round(as.numeric(con_ports_stats[idx_c_gmv, c("w1","w2","w3","w4","w5")]), 6),
  Con_Tan = round(as.numeric(con_ports_stats[idx_c_tan, c("w1","w2","w3","w4","w5")]), 6)
)
weights_table


############################################
# Question 2
############################################
# a
library(copula)
w_opt <- c(0.241983, 0.039362, 0.246873, 0.221792, 0.249990)  
# convert returns to uniform [0,1] via empirical CDF
u_data <- apply(ret_matrix, 2, function(x) rank(x) / (length(x) + 1))

# fit Gaussian copula
gauss_cop <- normalCopula(dim = ncol(ret_matrix), dispstr = "un")
fit_cop <- fitCopula(gauss_cop, data = u_data, method = "ml")

# extract estimated correlation matrix
rho_est <- getSigma(fit_cop@copula)
round(rho_est, 4)

# b
# simulate joint returns using the fitted copula
set.seed(2025)
n_sim <- 1e5
sim_u <- rCopula(n_sim, fit_cop@copula)  # simulate uniform variables

# transform back to returns using empirical quantiles
sim_ret <- matrix(NA, nrow = n_sim, ncol = ncol(ret_matrix))
for(i in 1:ncol(ret_matrix)){
  sim_ret[,i] <- quantile(ret_matrix[,i], probs = sim_u[,i], type = 1)
}

# portfolio returns
port_sim <- sim_ret %*% w_opt

# 95% 1-day VaR (negative)
VaR_copula <- -quantile(port_sim, 0.05)
VaR_copula

# variance–Covariance VaR
mu_p <- sum(colMeans(ret_matrix) * w_opt)
sigma_p <- sqrt(t(w_opt) %*% cov(ret_matrix) %*% w_opt)
VaR_varcov <- - (mu_p + sigma_p * qnorm(0.05))
VaR_varcov

# historical Simulation VaR
port_hist <- ret_matrix %*% w_opt
VaR_hist <- -quantile(port_hist, 0.05)
VaR_hist

# c
window <- 1000
alpha <- 0.10   # 90% confidence level
n_total <- nrow(ret_matrix)
VaR_roll <- rep(NA, n_total - window)

for(i in 1:(n_total - window)){
  print(i)
  in_sample <- ret_matrix[i:(i + window - 1), ]
  
  # fit Gaussian copula
  u_in <- apply(in_sample, 2, function(x) rank(x) / (length(x) + 1))
  cop_fit <- fitCopula(normalCopula(dim = ncol(in_sample), dispstr = "un"), u_in, method = "ml")
  
  # simulate
  sim_u <- rCopula(10000, cop_fit@copula)
  sim_ret <- matrix(NA, nrow = 10000, ncol = ncol(in_sample))
  for(j in 1:ncol(in_sample)){
    sim_ret[,j] <- quantile(in_sample[,j], probs = sim_u[,j], type = 1)
  }
  
  # portfolio returns
  port_sim <- sim_ret %*% w_opt
  
  # 1-day VaR at 90%
  VaR_roll[i] <- -quantile(port_sim, alpha)
}

# actual portfolio returns for comparison
port_actual <- ret_matrix[(window+1):n_total, ] %*% w_opt

# breaches
breaches <- port_actual < -VaR_roll
n_breaches <- sum(breaches)
expected_breaches <- length(VaR_roll) * alpha

# Binomial test
binom.test(n_breaches, length(VaR_roll), alpha)


###########################
# Question 3
###########################
library(rugarch)
w_gmv <- c(0.158339, 0.041610, 0.262412, 0.113070, 0.424570)  # Analytical GMV
w_tan <- c(-0.178259, -0.714753, 0.695231, -0.459548, 1.657329) # Analytical Tangency
port_gmv <- ret_matrix %*% w_gmv
port_tan <- ret_matrix %*% w_tan
# a

# GARCH(1,1) specification
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0,0)),
  distribution.model = "norm"
)

# fit GMV portfolio
fit_gmv <- ugarchfit(spec, port_gmv)
fit_gmv

# fit Tangency portfolio
fit_tan <- ugarchfit(spec, port_tan)
fit_tan

# extract conditional volatilities
vol_gmv <- sigma(fit_gmv)
vol_tan <- sigma(fit_tan)

df_vol <- tibble(
  Date = as.Date(1:length(vol_gmv)),  # replace with actual dates if available
  GMV = vol_gmv,
  Tangency = vol_tan
)


df_vol %>%
  pivot_longer(cols = -Date, names_to = "Portfolio", values_to = "Vol") %>%
  ggplot(aes(x = Date, y = Vol, color = Portfolio)) +
  geom_line() +
  labs(title = "Conditional Volatility (GARCH(1,1))", y = "Volatility", x = "Time") +
  theme_minimal()

# c
# optimal portfolio weights
w_opt <- c(0.241983, 0.039362, 0.246873, 0.221792, 0.249990)
port_opt <- ret_matrix %*% w_opt

# fit GARCH(1,1)
fit_opt <- ugarchfit(spec, port_opt)

# 1-day ahead forecast
forecast <- ugarchforecast(fit_opt, n.ahead = 1)
sigma_forecast <- sigma(forecast)  # conditional volatility

# 1-day 95% VaR (variance-covariance, normal)
alpha <- 0.05
VaR_garch <- - qnorm(alpha) * sigma_forecast
VaR_garch

# d
tibble(
  Method = c("GARCH(1,1) VaR", "Historical VaR", "Var-Cov VaR", "Copula VaR"),
  VaR_95 = c(VaR_garch, VaR_hist, VaR_varcov, VaR_copula)
)
## Round about the kernel
# Published 11/12/20

# Built using R 3.6.2

## Load packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidyquant)
  library(reticulate)
  library(generalCorr)
})

## Load data
prices_xts <- readRDS("corr_2_prices_xts.rds")

# Create function for rolling correlation
mean_cor <- function(returns)
{
  # calculate the correlation matrix
  cor_matrix <- cor(returns, use = "pairwise.complete")
  
  # set the diagonal to NA (may not be necessary)
  diag(cor_matrix) <- NA
  
  # calculate the mean correlation, removing the NA
  mean(cor_matrix, na.rm = TRUE)
}

# Create return frames for manipulation
comp_returns <- ROC(prices_xts[,-1], type = "discrete") # kernel regression
tot_returns <- ROC(prices_xts, type = "discrete") # for generalCorr

# Create data frame for regression
corr_comp <- rollapply(comp_returns, 60, 
                       mean_cor, by.column = FALSE, align = "right")

xli_rets <- ROC(prices_xts[,1], n=60, type = "discrete")

# Merge series and create train-test split
total_60 <- merge(corr_comp, lag.xts(xli_rets, -60))[60:(nrow(corr_comp)-60)]
colnames(total_60) <- c("corr", "xli")
split <- round(nrow(total_60)*.70)

train_60 <- total_60[1:split,]
test_60 <- total_60[(split+1):nrow(total_60),]

# Create train set for generalCorr
tot_split <- nrow(train_60)+60
train <- tot_returns[1:tot_split,]
test <- tot_returns[(tot_split+1):nrow(tot_returns),]

# Graph originaal scatter plot
train_60 %>% 
  ggplot(aes(corr*100, xli*100)) +
  geom_point(color = "darkblue", alpha = 0.4) +
  labs(x = "Correlation (%)",
       y = "Return (%)",
       title = "Return (XLI) vs. correlation (constituents)") +
  geom_smooth(method = "loess", formula = y ~ x, se=FALSE, size = 1.25, color = "red")


# Create helper function
cause_mat <- function(df){
  mat_1 <- df[,!apply(is.na(df),2, all)]
  mat_1 <- as.matrix(coredata(mat_1))
  out <- causeSummary(mat_1)
  out <- as.data.frame(out)
  out
}

# Create column and row indices
col_idx <- list(c(1:22), c(1,23:44), c(1,45:64))
row_idx <- list(c(1:250), c(251:500), c(501:750), c(751:1000),
                c(1001:1250), c(1251:1500), c(1501:1750), c(1751:2000),
                c(2001:2250), c(2251:2500))

# Create cause list for each period: which stocks cause the index
cause <- list()
for(i in 1:length(row_idx)){
  out <- list()
  for(j in 1:length(col_idx)){
    out[[j]] <- cause_mat(train[row_idx[[i]], col_idx[[j]]])
  }
  cause[[i]] <- out
}


# Bind cause into one list
cause_lists <- list()

for(i in 1:length(cause)){
  out <- do.call("rbind", cause[[i]]) %>%
    filter(cause != "xli") %>%
    select(cause) %>%
    unlist() %>%
    as.character()

  cause_lists[[i]] <- out

}

# Save cause_lists for use in Python
max_l <- 0
for(i in 1:length(cause_lists)){
  if(length(cause_lists[[i]]) > max_l){
    max_l <- length(cause_lists[[i]])
  }
}

write_l <- matrix(nrow = length(cause_lists), ncol = max_l)
for(i in 1:length(cause_lists)){
  write_l[i, 1:length(cause_lists[[i]])] <- cause_lists[[i]]
           
}

write.csv(write_l, "cause_lists.csv")



## Use cause list to run rolling correlations and aggregate forward returns for regression
cor_idx <- list(c(191:500), c(441:750), c(691:1000), c(941:1250),
                c(1191:1500), c(1441:1750), c(1691:2000), c(1941:2250),
                c(2191:2500))

# Add 1 since xli is price while train is ret so begin date is off by 1 biz day
ret_idx <- list(c(251:561), c(501:811), c(751:1061), c(1001:1311),
                c(1251:1561), c(1501:1811), c(1751:2061), c(2001:2311),
                c(2251:2561))


merge_list <- list()
for(i in 1:length(cor_idx)){
  corr <- rollapply(train[cor_idx[[i]], cause_lists[[i]]], 60,
                    mean_cor, by.column = FALSE, align = "right")
  ret <- ROC(prices_xts[ret_idx[[i]],1], n=60, type = "discrete")
  merge_list[[i]] <- merge(corr = corr[60:310], xli = coredata(ret[61:311]))
}

# Run correlations on non cause list
non_cause_list <- list()
for(i in 1:length(cor_idx)){
  corr <- rollapply(train[cor_idx[[i]], !colnames(train)[-1] %in% cause_lists[[i]]], 60,
                    mean_cor, by.column = FALSE, align = "right")
  ret <- ROC(prices_xts[ret_idx[[i]],1], n=60, type = "discrete")
  non_cause_list[[i]] <- merge(corr = corr[60:310], xli = coredata(ret[61:311]))
}


## Load data
merge_list <- readRDS("corr3_genCorr_list.rds")
non_cause_list <- readRDS("corr3_genCorr_non_cause_list.rds")

# Graphical example of one period
cause_ex <- merge_list[[3]]

cause_ex$corr_non <- rollapply(train[cor_idx[[3]], !colnames(train)[-1] %in% cause_lists[[3]]],
                                     60, mean_cor, by.column = FALSE, align = "right")[60:310]

# Graph causal subset against returns
cause_ex %>%
  ggplot(aes(corr*100, xli*100)) +
  geom_point(color = "blue") +
  geom_smooth(method="lm", formula = y ~ x, se=FALSE, color = "darkgrey", linetype = "dashed")+
  geom_smooth(method="loess", formula = y ~ x, se=FALSE, color = "darkblue") +
  labs(x = "Correlation (%)",
       y = "Return (%)",
       title = "Return (XLI) vs. correlation (causal subset)")

# Graph non causal
cause_ex %>%
  ggplot(aes(corr_non*100, xli*100)) +
  geom_point(color = "blue") +
  geom_smooth(method="lm", formula = y ~ x, se=FALSE, color = "darkgrey", linetype = "dashed")+
  geom_smooth(method="loess", formula = y ~ x, se=FALSE, color = "darkblue") +
  labs(x = "Correlation (%)",
       y = "Return (%)",
       title = "Return (XLI) vs. correlation (non-causal subset)")

# Run models
causal_kern <- kern(cause_ex$xli, cause_ex$corr)$R2
causal_lin <- summary(lm(cause_ex$xli ~ cause_ex$corr))$r.squared
non_causal_kern <- kern(cause_ex$xli, cause_ex$corr_non)$R2
non_causal_lin <- summary(lm(cause_ex$xli ~ cause_ex$corr_non))$r.squared

# Show table
data.frame(Models = c("Kernel", "Linear"), 
           Causal = c(causal_kern, causal_lin),
           `Non-causal` = c(non_causal_kern, non_causal_lin),
           check.names = FALSE) %>% 
  mutate_at(vars('Causal', `Non-causal`), function(x) round(x,3)*100) %>% 
  knitr::kable(caption = "Regression R-squareds (%)")

## Linear regression
models <- list()
for(i in 1:length(merge_list)){
  models[[i]] <- lm(xli~corr, merge_list[[i]])
}

model_df <- data.frame(model = seq(1,length(models)),
                       rsq = rep(0,length(models)),
                       t_int = rep(0,length(models)),
                       t_coef = rep(0,length(models)),
                       P_int = rep(0,length(models)),
                       p_coef = rep(0,length(models)))

for(i in 1:length(models)){
  model_df[i,2] <- broom::glance(models[[i]])[1]
  model_df[i,3] <- broom::tidy(models[[i]])[1,4]
  model_df[i,4] <- broom::tidy(models[[i]])[2,4]
  model_df[i,5] <- broom::tidy(models[[i]])[1,5]
  model_df[i,6] <- broom::tidy(models[[i]])[2,5]
}

start <- index(train)[seq(250,2250,250)] %>% year()
end <- index(train)[seq(500,2500,250)] %>% year()

model_dates <- paste(start, end, sep = "-")

model_df <- model_df %>%
  mutate(model_dates = model_dates) %>%
  select(model_dates, everything())


## Kernel regresssion
kernel_models <- list()
for(i in 1:length(merge_list)){
  kernel_models[[i]] <- kern(merge_list[[i]]$xli, merge_list[[i]]$corr)
}


kern_model_df <- data.frame(model_dates = model_dates,
                       rsq = rep(0,length(kernel_models)),
                       rmse = rep(0,length(kernel_models)),
                       rmse_scaled = rep(0,length(kernel_models)))

for(i in 1:length(kernel_models)){
  kern_model_df[i,2] <- kernel_models[[i]]$R2
  kern_model_df[i,3] <- sqrt(kernel_models[[i]]$MSE)
  kern_model_df[i,4] <- sqrt(kernel_models[[i]]$MSE)/sd(merge_list[[i]]$xli)
}

## Load data
model_df <- readRDS("corr3_lin_model_df.rds")
kern_model_df <- readRDS("corr3_kern_model_df.rds")

## R-squared graph
data.frame(Dates = model_dates, 
           Linear = model_df$rsq,
           Kernel = kern_model_df$rsq) %>% 
  gather(key, value, -Dates) %>% 
  ggplot(aes(Dates, value*100, fill = key)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual("", values = c("blue", "darkgrey")) +
  labs(x = "",
       y = "R-squared (%)",
       title = "R-squared output for regression results by period and model") +
  theme(legend.position = c(0.06,0.9),
        legend.background = element_rect(fill = NA)) 

# NOn_causal linear model
non_models <- list()
for(i in 1:length(reg_list)){
  non_models[[i]] <- lm(xli~corr, non_cause_list[[i]])
}

non_model_df <- data.frame(model = seq(1,length(models)),
                           rsq = rep(0,length(models)),
                           t_int = rep(0,length(models)),
                           t_coef = rep(0,length(models)),
                           P_int = rep(0,length(models)),
                           p_coef = rep(0,length(models)))

for(i in 1:length(non_models)){
  non_model_df[i,2] <- broom::glance(non_models[[i]])[1]
  non_model_df[i,3] <- broom::tidy(non_models[[i]])[1,4]
  non_model_df[i,4] <- broom::tidy(non_models[[i]])[2,4]
  non_model_df[i,5] <- broom::tidy(non_models[[i]])[1,5]
  non_model_df[i,6] <- broom::tidy(non_models[[i]])[2,5]
}

non_model_df <- non_model_df %>%
  mutate(model_dates = model_dates) %>%
  select(model_dates, everything())

# Bar chart of causal and non-causal
data.frame(Dates = model_dates, 
           `Linear--causal` = model_df$rsq,
           `Linear--non-causal` = non_model_df$rsq,
           Kernel = kern_model_df$rsq,
           check.names = FALSE) %>% 
  gather(key, value, -Dates) %>% 
  ggplot(aes(Dates, value*100, fill = key)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual("", values = c("blue", "darkgrey", "darkblue")) +
  labs(x = "",
       y = "R-squared (%)",
       title = "R-squared output for regression results by period and model") +
  theme(legend.position = c(0.3,0.9),
        legend.background = element_rect(fill = NA)) 

## RMSE comparison
lin_rmse <- c()
lin_non_rmse <- c()
kern_rmse <- c()

for(i in 1:length(models)){
  lin_rmse[i] <- sqrt(mean(models[[i]]$residuals^2))
  lin_non_rmse[i] <- sqrt(mean(non_models[[i]]$residuals^2))
  kern_rmse[i] <- sqrt(kernel_models[[i]]$MSE)
}

data.frame(Dates = model_dates, 
           `Linear--causal` = lin_rmse,
           `Linear--non-causal` = lin_non_rmse,
           Kernel = kern_rmse,
           check.names = FALSE) %>% 
  gather(key, value, -Dates) %>% 
  ggplot(aes(Dates, value*100, fill = key)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual("", values = c("blue", "darkgrey", "darkblue")) +
  labs(x = "",
       y = "RMSE (%)",
       title = "RMSE results by period and model") +
  theme(legend.position = c(0.08,0.9),
        legend.background = element_rect(fill = NA)) 

## RMSE graph
data.frame(Dates = model_dates, 
           `Kernel - Linear-causal` = lin_rmse - kern_rmse,
           `Kernel - Linear--non-causal` = lin_non_rmse - kern_rmse ,
           check.names = FALSE) %>% 
  gather(key, value, -Dates) %>% 
  ggplot(aes(Dates, value*100, fill = key)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual("", values = c("darkgrey", "darkblue")) +
  labs(x = "",
       y = "RMSE (%)",
       title = "RMSE differences by period and model") +
  theme(legend.position = c(0.1,0.9),
        legend.background = element_rect(fill = NA))

avg_lin <- round(mean(lin_rmse - kern_rmse),3)*100
avg_lin_non <- round(mean(lin_non_rmse - kern_rmse),3)*100

## Price graph
prices_xts["2010/2014","xli"] %>%  
  ggplot(aes(index(prices_xts["2010/2014"]), xli)) +
  geom_line(color="blue", size = 1.25) +
  labs(x = "",
       y = "Price (US$)",
       title = "XLI price log-scale") +
  scale_y_log10()
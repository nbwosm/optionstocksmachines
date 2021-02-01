# Portfolio simulations

## Portfolio simuation function
port_sim <- function(df, sims, cols){
  
  if(ncol(df) != cols){
    print("Columns don't match")
    break
  }  
  
  # Create weight matrix
  wts <- matrix(nrow = sims, ncol = cols)
  
  for(i in 1:sims){
    a <- runif(cols,0,1)
    b <- a/sum(a)
    wts[i,] <- b
  }
  
  # Find returns
  mean_ret <- colMeans(df)
  
  # Calculate covariance matrix
  cov_mat <- cov(df)
  
  # Calculate random portfolios
  port <- matrix(nrow = sims, ncol = 2)
  for(i in 1:sims){
    port[i,1] <- as.numeric(sum(wts[i,] * mean_ret))
    port[i,2] <- as.numeric(sqrt(t(wts[i,]) %*% cov_mat %*% wts[i,]))
  }
  
  colnames(port) <- c("returns", "risk")
  port <- as.data.frame(port)
  port$sharpe <- port$returns/port$risk*sqrt(12)
  
  max_sharpe <- port[which.max(port$sharpe),]
  
  graph <- port %>% 
    ggplot(aes(risk*sqrt(12)*100, returns*1200, color = sharpe)) +
    geom_point(size = 1.2, alpha = 0.4) +
    scale_color_gradient(low = "darkgrey", high = "darkblue") +
    labs(x = "Risk (%)",
         y = "Return (%)",
         title = "Simulated portfolios")
  
  out <- list(port = port, graph = graph, max_sharpe = max_sharpe, wts = wts)
  
}

## Portfolio Simulation leave 
port_sim_lv <- function(df, sims, cols){
  
  if(ncol(df) != cols){
    print("Columns don't match")
    break
  }  
  
  # Create weight matrix
  wts <- matrix(nrow = (cols-1)*sims, ncol = cols)
  count <- 1
  
  for(i in 1:(cols-1)){
    for(j in 1:sims){
      a <- runif((cols-i+1),0,1)
      b <- a/sum(a)
      c <- sample(c(b,rep(0,i-1)))
      wts[count,] <- c
      count <- count+1
    }
    
  }
  
  # Find returns
  mean_ret <- colMeans(df)
  
  # Calculate covariance matrix
  cov_mat <- cov(df)
  
  # Calculate random portfolios
  port <- matrix(nrow = (cols-1)*sims, ncol = 2)
  for(i in 1:nrow(port)){
    port[i,1] <- as.numeric(sum(wts[i,] * mean_ret))
    port[i,2] <- as.numeric(sqrt(t(wts[i,]) %*% cov_mat %*% wts[i,]))
    
  }
  
  colnames(port) <- c("returns", "risk")
  port <- as.data.frame(port)
  port$sharpe <- port$returns/port$risk*sqrt(12)
  
  max_sharpe <- port[which.max(port$sharpe),]
  
  graph <- port %>% 
    ggplot(aes(risk*sqrt(12)*100, returns*1200, color = sharpe)) +
    geom_point(size = 1.2, alpha = 0.4) +
    scale_color_gradient(low = "darkgrey", high = "darkblue") +
    labs(x = "Risk (%)",
         y = "Return (%)",
         title = "Simulated portfolios")
  
  out <- list(port = port, graph = graph, max_sharpe = max_sharpe, wts = wts)
  
}

## Load portfolio selection function
port_select_func <- function(port, return_min, risk_max, port_names){
  port_select  <-  cbind(port$port, port$wts)
  
  port_wts <- port_select %>% 
    mutate(returns = returns*12,
           risk = risk*sqrt(12)) %>% 
    filter(returns >= return_min,
           risk <= risk_max) %>% 
    summarise_at(vars(4:7), mean) %>% 
    `colnames<-`(port_names)
  
  p <- port_wts %>%
    rename("Stocks" = stock,
           "Bonds" = bond,
           "Gold" = gold,
           "Real estate" = realt) %>% 
    gather(key,value) %>% 
    ggplot(aes(reorder(key,value), value*100 )) +
    geom_bar(stat='identity', position = "dodge", fill = "blue") +
    geom_text(aes(label=round(value,2)*100), vjust = -0.5) +
    scale_y_continuous(limits = c(0,max(port_wts*100+2))) +
    labs(x="",
         y = "Weights (%)",
         title = "Average weights for risk-return constraints")
  
  out <- list(port_wts = port_wts, graph = p)
  
  out
  
}


## Function for portfolio returns without rebalancing
rebal_func <- function(act_ret, weights){
  ret_vec <- c()
  wt_mat <- matrix(nrow = nrow(act_ret), ncol = ncol(act_ret))
  for(i in 1:nrow(wt_mat)){
    wt_ret <- act_ret[i,]*weights # wt'd return
    ret <- sum(wt_ret) # total return
    ret_vec[i] <- ret 
    weights <- (weights + wt_ret)/(sum(weights)+ret) # new weight based on change in asset value
    wt_mat[i,] <- as.numeric(weights)
  }
  out <- list(ret_vec = ret_vec, wt_mat = wt_mat)
  out
}


## Function for calculating satisfactory weighting 
port_sim_wts <- function(df, sims, cols, return_min, risk_max){
  
  if(ncol(df) != cols){
    print("Columns don't match")
    break
  }  
  
  # Create weight matrix
  wts <- matrix(nrow = (cols-1)*sims, ncol = cols)
  count <- 1
  
  for(i in 1:(cols-1)){
    for(j in 1:sims){
      a <- runif((cols-i+1),0,1)
      b <- a/sum(a)
      c <- sample(c(b,rep(0,i-1)))
      wts[count,] <- c
      count <- count+1
    }
    
  }
  
  # Find returns
  mean_ret <- colMeans(df)
  
  # Calculate covariance matrix
  cov_mat <- cov(df)
  
  # Calculate random portfolios
  port <- matrix(nrow = (cols-1)*sims, ncol = 2)
  for(i in 1:nrow(port)){
    port[i,1] <- as.numeric(sum(wts[i,] * mean_ret))
    port[i,2] <- as.numeric(sqrt(t(wts[i,]) %*% cov_mat %*% wts[i,]))
    
  }
  
  port <- as.data.frame(port) %>% 
    `colnames<-`(c("returns", "risk"))
  
  
  port_select  <-  cbind(port, wts)
  
  port_wts <- port_select %>% 
    mutate(returns = returns*12,
           risk = risk*sqrt(12)) %>% 
    filter(returns >= return_min,
           risk <= risk_max) %>% 
    summarise_at(vars(3:6), mean)
  
  port_wts
  
}


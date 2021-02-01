# Efficient frontier

eff_frontier <- function(returns, short = FALSE, max_allocation = NULL, risk_premium_up = 0.5, 
                         risk_increment = 0.005){
  covariance <- cov(returns)
  num <- ncol(covariance)
  
  if(short){
    Amat <- matrix(1,nrow = num)
    bvec <- 1
    meq <- 1
  }else{
    Amat <- cbind(1, diag(num))
    bvec <- c(1, rep(0, num))
    meq <- 1
  }
  
  if(!is.null(max_allocation)){
    if(max_allocation > 1 | max_allocation < 0){
      stop("Max_allocation must be greater than zero or less than one.")
    }else if(max_allocation * num < 1){
      stop("Max_allocation must be set higher so that enough assets sum to one.")
    } else{
    Amat <- cbind(Amat, -diag(num))
    bvec <- c(bvec, rep(-max_allocation, n))
    }
  }
  
  loops <- risk_premium_up/risk_increment+1
  loop <- 1
  
  eff <- matrix(nrow = loops, ncol = num + 3)
  colnames(eff) <- c(colnames(returns), "stdev", "exp_ret", "sharpe")
  
  L <- seq(0, risk_premium_up, risk_increment)
  
  for(i in L){
    dvec <- colMeans(returns)*i
    sol <- quadprog::solve.QP(covariance, dvec = dvec, Amat = Amat, bvec = bvec, meq = meq)
    eff[loop, "stdev"] <- sqrt(sum(sol$solution * colSums(covariance * sol$solution)))
    eff[loop, "exp_ret"] <- as.numeric(sol$solution %*% colMeans(returns))
    eff[loop, "sharpe"] <- eff[loop,"exp_ret"]/eff[loop, "stdev"]
    eff[loop, 1:num] <- sol$solution
    loop <- loop + 1
  }
  return(as.data.frame(eff))
  
}


eff_frontier_long <- function(returns, risk_premium_up = 0.5, risk_increment = 0.005){
  covariance <- cov(returns, use = "pairwise.complete.obs")
  num <- ncol(covariance)
  
  Amat <- cbind(1, diag(num))
  bvec <- c(1, rep(0, num))
  meq <- 1
  
  risk_steps <- risk_premium_up/risk_increment+1
  count <- 1
  
  eff <- matrix(nrow = risk_steps, ncol = num + 3)
  colnames(eff) <- c(colnames(returns), "stdev", "exp_ret", "sharpe")
  
  loop_step <- seq(0, risk_premium_up, risk_increment)
  
  for(i in loop_step){
    dvec <- colMeans(returns, na.rm = TRUE)*i
    sol <- quadprog::solve.QP(covariance, dvec = dvec, Amat = Amat, bvec = bvec, meq = meq)
    eff[count, "stdev"] <- sqrt(sum(sol$solution * colSums(covariance * sol$solution)))
    eff[count, "exp_ret"] <- as.numeric(sol$solution %*% colMeans(returns, na.rm = TRUE))
    eff[count, "sharpe"] <- eff[count,"exp_ret"]/eff[count, "stdev"]
    eff[count, 1:num] <- sol$solution
    count <- count + 1
  }
  return(as.data.frame(eff))
  
}

max_sharpe <- function(returns, risk_premium_up = 0.5, risk_increment = 0.005){
  covariance <- cov(returns, use = "pairwise.complete.obs")
  num <- ncol(covariance)
  
  Amat <- cbind(1, diag(num))
  bvec <- c(1, rep(0, num))
  meq <- 1
  
  risk_steps <- risk_premium_up/risk_increment+1
  count <- 1
  
  eff <- matrix(nrow = risk_steps, ncol = num + 3)
  colnames(eff) <- c(colnames(returns), "stdev", "exp_ret", "sharpe")
  
  loop_step <- seq(0, risk_premium_up, risk_increment)
  
  for(i in loop_step){
    dvec <- colMeans(returns, na.rm = TRUE)*i
    sol <- quadprog::solve.QP(covariance, dvec = dvec, Amat = Amat, bvec = bvec, meq = meq)
    eff[count, "stdev"] <- sqrt(sum(sol$solution * colSums(covariance * sol$solution)))
    eff[count, "exp_ret"] <- as.numeric(sol$solution %*% colMeans(returns, na.rm = TRUE))
    eff[count, "sharpe"] <- eff[count,"exp_ret"]/eff[count, "stdev"]
    eff[count, 1:num] <- sol$solution
    count <- count + 1
  }
  return(as.data.frame(eff))
  
}

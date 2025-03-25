
library(tidyverse)
library(data.table)
library(lubridate)
library(caret)
library(ggsci)
# library(ggpmisc)
library(ggpubr)
library(readxl)
library(nls.multstart)
library(quantreg)
library(progress)

# Function to extract temperature course for specific site and time interval
extract_covar <- function(df, from, to) {
  df_ <- df %>% filter(timestamp > from, timestamp <= to)
  return(df_)
}

extract_covars_from_nested <- function(tbl, from, vars)
{
  dat_vars <- list()
  for(i in vars){
    dat_vars[[i]] <- do.call("c", lapply(tbl[[from]], "[[", i))
  }
  if(length(vars)>1){
    vars <- do.call("cbind", dat_vars)
    out <- cbind(tbl, vars)
    out <- as_tibble(out)
  } else {
    out <- as_tibble(dat_vars)
    out <- bind_cols(tbl, out)
  }
  return(out)
}

# calculate GDD
calculate_gdd <- function(data, t_base, t_max){
  
  if (nrow(data) == 0) return(0)
  
  # Calculate hourly GDD with temperature clipping
  gdd <- data %>%
    mutate(
      temp_clipped = pmin(pmax(temp - t_base, 0), t_max - t_base), # Cap temperatures
      time_diff = as.numeric(difftime(lead(timestamp, default = last(timestamp)), timestamp, units = "hours")) / 24, # Convert hours to days
      GDD_contribution = temp_clipped * time_diff)
  # Sum up daily GDD
  total_gdd <- sum(gdd$GDD_contribution, na.rm = T)
  
  return(total_gdd)
  
}

# calculate GDD
calculate_gdd2 <- function(data, t_base, t_max, t1, t2){
  
  if (nrow(data) == 0) return(0)
  
  # Calculate hourly GDD with temperature clipping
  gdd <- data %>%
    mutate(
      temp_clipped = pmin(pmax(temp - t_base, 0), t_max - t_base), # Cap temperatures
      time_diff = as.numeric(difftime(.data[[t1]], .data[[t2]], units = "hours")) / 24, # Convert hours to days
      GDD_contribution = temp_clipped * time_diff)
  # Sum up daily GDD
  total_gdd <- sum(gdd$GDD_contribution, na.rm = T)
  
  return(total_gdd)
  
}

get_lags_diffs <- function(data, vars) {
  
  # get lag values for all lesion features
  df_lags <- data %>%
    mutate(across(
      .cols = vars, 
      .fns = list(lag_ = ~lag(.)), 
      .names = "lag_{.col}"
    ))
  
  # get relevant differences
  df_lags <- df_lags %>% mutate(lag_timestamp = dplyr::lag(timestamp)) %>% 
    relocate(lag_timestamp, .after=timestamp)
  df_lags_deltas <- df_lags %>% 
    mutate(
      diff_time = difftime(timestamp, lag_timestamp, units = "hours"),
      diff_gdd = lesion_age_gdd - lag_lesion_age_gdd,
      diff_area = area - lag_area,
      # perimeter-normalized differences 
      diff_area_pp_xy = diff_area / lag_xy_perimeter_f,
      diff_area_pp_x = diff_area / lag_x_perimeter_f,
      diff_area_pp_y = diff_area / lag_y_perimeter_f
    )
  
  return(df_lags_deltas)
  
}

exp_decay <- function(y0, yf, alpha, t) {
  y <- yf + (y0 - yf) * exp(-alpha * t)
  return(y)
}

# log-transformed exponential
nls_exp_log <- function(data){
  fit <- nls_multstart(log(diff_area_norm_gdd_peri_y) ~ exp_decay(y0, yf, alpha, t = lesion_age_gdd),
                       data = data, 
                       iter = 200, 
                       start_lower = c(y0 = -3, yf = -4, alpha = 0),  # lower starting value bounds
                       start_upper = c(y0 = 0, yf = -1, alpha = 1),  # upper starting value bounds
                       convergence_count = 150,
                       supp_errors = "Y",
                       lower = c(-3, -4, 0),  # lower bounds on each parameter (equals lower starting bounds, arbitrary choice!)
                       upper = c(0, -1, 1))  # upper bounds on each parameter (equals upper starting bounds, arbitrary choice!)
}

# un-transformed exponential
nls_exp <- function(data){
  fit <- nls_multstart(diff_area_norm_gdd_peri_y ~ exp_decay(y0, yf, alpha, t = lesion_age_gdd),
                       data = data, 
                       iter = 200, 
                       start_lower = c(y0 = 0, yf = 0, alpha = 0),  # lower starting value bounds
                       start_upper = c(y0 = 0.1, yf = 0.1, alpha = 10),  # upper starting value bounds
                       convergence_count = 150,
                       supp_errors = "Y",
                       lower = c(0.1, 0, 0),  # lower bounds on each parameter (equals lower starting bounds, arbitrary choice!)
                       upper = c(3, 0.1, 10))  # upper bounds on each parameter (equals upper starting bounds, arbitrary choice!)
}

# un-transformed double exponential
double_exp_decay <- function(y0, yf, alpha1, alpha2, t, w) {
  y <- yf + (y0 - yf) * (w * exp(-alpha1 * t) + (1 - w) * exp(-alpha2 * t))
  return(y)
}

nls_double_exp <- function(data){
  # pb$tick()
  fit <- nls_multstart(diff_area_norm_gdd_peri_y ~ double_exp_decay(y0, yf, alpha1, alpha2, w, t = lesion_age_gdd),
                       data = data, 
                       iter = 200, 
                       start_lower = c(y0 = 0.1, yf = 0, alpha1 = 0.1, alpha2 = 0.01, w = 0.5),  # lower starting value bounds
                       start_upper = c(y0 = 1, yf = 0.1, alpha1 = 1, alpha2 = 0.5, w = 1),  # upper starting value bounds
                       convergence_count = 150,
                       supp_errors = "Y",
                       lower = c(0.1, -0.2, 0.1, 0, 0.2),  # lower bounds on each parameter (equals lower starting bounds, arbitrary choice!)
                       upper = c(3, 0.2, 1.5, 0.8, 1.2))  # upper bounds on each parameter (equals upper starting bounds, arbitrary choice!)
}

# quantile regression 
nls_quantile_exp <- function(data, x, y, tau = 0.5, n_samples = 300) {
  
  n_samples <- n_samples
  start_samples <- data.frame(
    yf = runif(n_samples, min = 0, max = .01),
    alpha = runif(n_samples, min = 0, max = 1)
  ) %>% 
    mutate(y0 = runif(n_samples, min = yf, max = yf + .1))
  
  formula = as.formula(paste0(y, "~exp_decay(y0, yf, alpha, ", x, ")"))
  
  results <- start_samples %>%
    rowwise() %>%
    mutate(
      fit = list(tryCatch(
        fit <- nlrq(formula, data = data, tau = tau,
                    start = list(y0 = y0, yf = yf, alpha = alpha)),
        error = function(e) NULL
      ))
    )
  
  r <- results %>%
    filter(!is.null(fit)) %>% 
    mutate(deviance = compute_deviance(fit, data = data, tau = 0.5)) %>% 
    ungroup() %>% 
    mutate(pars = purrr::map(.x = fit, .f = get_qr_pars)) %>% unnest(pars) %>% 
    filter(alpha_fitted > 0) %>% 
    filter(y0_fitted >= 0) %>% 
    filter(alpha_fitted <= 1) %>% 
    filter(yf_fitted >= 0)
  
  best_fit <- r %>% 
    slice_min(deviance) %>%
    pull(fit)
  
  if (length(best_fit) > 0){
    return(best_fit[[1]])
    
  } else{
    return(NA)
  }
  
  # # tau is the desired quantile (e.g., 0.5 for the median)
  # fit <- nlrq(diff_area_norm_gdd_peri_y ~ exp_decay(y0, yf, alpha, t = lesion_age_gdd),
  #             data = data,
  #             tau = tau,
  #             start = list(y0 = 0.01, yf = 0, alpha = 0.001))
  # return(fit)
}

compute_deviance <- function(fit, data, tau = 0.5) {
  if (is.null(fit)) return(NA)
  
  # Observed and predicted values
  y <- data$diff_area_norm_gdd_peri_y
  y_hat <- unlist(predict(fit, newdata = data))
  
  # Residuals
  residuals <- y - y_hat
  
  # Weighted absolute residuals
  weighted_residuals <- ifelse(residuals > 0, tau * abs(residuals), (1 - tau) * abs(residuals))
  
  # Sum of weighted residuals
  sum(weighted_residuals, na.rm = TRUE)
}

# try with quantile regression
nls_quantile_double_exp <- function(data, tau = 0.5) {
  
  n_samples <- 200 # Number of random starts
  start_samples <- data.frame(
    y0 = runif(n_samples, min = 0, max = .1),
    yf = runif(n_samples, min = 0, max = .01),
    alpha1 = runif(n_samples, min = .5, max = 1),
    alpha2 = runif(n_samples, min = .01, max = .1),
    w = runif(n_samples, min = .1, max = .9)
  )
  
  results <- start_samples %>%
    rowwise() %>%
    mutate(
      fit = list(tryCatch(
        nlrq(diff_area_norm_gdd_peri_y ~ double_exp_decay(y0, yf, alpha1, alpha2, w, t = lesion_age_gdd),
             data = data,
             tau = 0.5,
             start = list(y0 = y0, yf = yf, alpha1 = alpha1, alpha2 = alpha2, w = w)),
        error = function(e) NULL
      ))
    )
  
  results <- results %>%
    mutate(deviance = compute_deviance(fit, data = data, tau = 0.5))
  
  best_fit <- results %>% ungroup() %>% 
    slice_min(deviance) %>%
    pull(fit)
  
  return(best_fit[[1]])
  
  summary(best_fit[[1]])
  
#   # Generate predictions
#   data_with_predictions <- data %>%
#     mutate(
#       predicted = predict(best_fit[[1]], newdata = data)
#     )
#   
#   # Create the plot
#   ggplot(data_with_predictions, aes(x = lesion_age_gdd, y = diff_area_norm_gdd_peri_y)) +
#     geom_point(color = "blue", alpha = 0.6) + # Observed data points
#     geom_line(aes(y = predicted), color = "red", size = 1) + # Predicted values
#     labs(
#       title = "Observed Data and Best Fit",
#       x = "Lesion Age (GDD)",
#       y = "Normalized Area Difference"
#     ) +
#     theme_minimal()
# }
#   
#   fit <- nlrq(diff_area_norm_gdd_peri_y ~ double_exp_decay(y0, yf, alpha1, alpha2, w, t = lesion_age_gdd),
#               data = data,
#               tau = tau,
#               start = list(y0 = .1, yf = 0, alpha1 = 1.5, alpha2 = 0.025, w = 0.1),
#               trace = TRUE)
#   return(fit)
}

# quantile regression 
linear_quantile <- function(data, x, y) {
  formula = as.formula(paste0(y, "~", x))
  fit <- rq(formula, data = data)
  return(fit)
}


tidy_model_output <- function(data){
  out <- data[,c("term", "estimate")] %>% 
    tidyr::spread(term, estimate)
  return(out)
}

get_lm_pars <- function(obj)
{
  slope <- unname(coef(obj)[2])
  intercept <- unname(coef(obj)["(Intercept)"])
  tibble(slope, intercept)
}

get_qr_pars <- function(obj)
{
  if (is.null(obj)) return(NA)
  coefs <- coef(obj)
  t <- tibble(!!!coef(obj))
  names(t) <- paste0(names(t), "_fitted")
  return(t)
}

get_RMSE <- function(obj)
{
  RSS <- c(crossprod(obj$residuals))
  MSE <- RSS / length(obj$residuals)
  RMSE <- sqrt(MSE)
}



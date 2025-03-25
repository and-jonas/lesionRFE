
library(tidyverse)

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

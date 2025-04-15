
#====================================================================================== -

#HEADER ----

# Author: Jonas Anderegg, ETH Zürich
# Copyright (C) 2025  ETH Z?rich, Jonas Anderegg (jonas.anderegg@usys.ethz.ch)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#  
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

#====================================================================================== -

rm(list = ls())

# .libPaths("T:/R4Userlibs")
# .libPaths("E:/Rlibs")
.libPaths("/home/anjonas/R4Userlibs")

list.of.packages <- c("data.table", "doParallel", "caret", "tidyverse", "ranger", "ggplot2", "ggpubr", "foreach", "tidyverse", "corrplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib = "/home/anjonas/R4Userlibs", dependencies = TRUE, repos='https://stat.ethz.ch/CRAN/')

library(caret)
library(data.table)
library(ranger)
library(tidyverse)
library(ggpubr)
library(doParallel)
library(doSNOW)
library(foreach)
library(corrplot)
path_to_utils <- "/projects/lesionRFE/" 
source(paste0(path_to_utils, "rfe_PAR.R"))
source(paste0(path_to_utils, "utils.R"))

# define scenario
target_trait <- "diff_area_pp_y_norm_chr"

# set paths
base_output_path <- paste0("/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/Results/RFE_", target_trait, "_test")
figure_path <- paste0(base_output_path, "/Figures/")
path_input_data <- "/home/anjonas/public/Public/Jonas/011_STB_leaf_tracking/data/subset_step2.rds"
run_paths <- paste0(base_output_path, c("/run0", "/run1", "/run2"))

# make output directories
for (d in c(base_output_path, figure_path, run_paths)){
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}

# =================================================================================================== -
# Prepare feature data ----
# =================================================================================================== -

# load training data
data0 <- readRDS(path_input_data)

# get relevant variables
data <- data0 %>% 
  extract_covars_from_nested(., from = "design", vars = "genotype_name") %>%
  dplyr::select(all_of(target_trait), exp_UID, genotype_name, batch, lesion_nr, diff_time, diff_gdd,  
                mean_interval_temp, cv_interval_temp, mean_interval_rh, cv_interval_rh, contains("lag_")) %>% 
  dplyr::select(-lag_timestamp, -lag_timestamp_leaf,
                -lag_perimeter, -lag_analyzable_perimeter, -lag_occluded_perimeter) %>% 
  dplyr::filter(complete.cases(.))

# drop redundant variables
data <- data %>%
  dplyr::select(-lag_max_p_density) # redundant to lag_max_l_density

# adjust data types
data <- data %>% 
  # two factors
  mutate(across(c(exp_UID, genotype_name, batch), as.factor)) %>% 
  # rest is numeric
  mutate(across(c(lesion_nr:lag_lesion_age_gdd), as.numeric))

# =================================================================================================== -
# Plot feature data ----
# =================================================================================================== -

# plot training data (numeric variables only)
long <- data %>%
  pivot_longer(cols = lesion_nr:lag_lesion_age_gdd, names_to = "feature", values_to = "value")

scaling_function <- function(X) (X - min(X, na.rm = TRUE))/diff(range(X, na.rm = TRUE))*1
scaledat <-  long %>% group_by(feature) %>% group_nest() %>% 
  mutate(scaled_val = purrr::map(data,~scaling_function(.$value))) %>% 
  unnest(cols = c(data, scaled_val)) %>% ungroup()

plot <- ggplot(scaledat) +
  geom_density(aes(x=scaled_val, group = exp_UID, fill = exp_UID), alpha = 0.2) +
  scale_x_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1)) +
  facet_wrap(~feature, scales = "free_y") +
  ylab("Density") + xlab("Scaled feature value") +
  theme_bw() +
  theme(
    panel.border = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(face="bold", size = 5)
  )

png(paste0(figure_path, "variable_distr_nolag.png"),
    width = 12, height = 7, units = 'in', res = 600)
plot(plot)
dev.off()

# UNSCALED
plot <- ggplot(long) +
  geom_density(aes(x=value, group = exp_UID, fill = exp_UID), alpha = 0.2) +
  facet_wrap(~feature, scales = "free") +
  ylab("Density") + xlab("Scaled feature value") +
  theme_bw() +
  theme(
    panel.border = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.background = element_blank(),
    strip.text = element_text(face="bold", size = 10)
  )
png(paste0(figure_path, "variable_distr_nolag_noscale.png"),
    width = 18, height = 10.5, units = 'in', res = 600)
plot(plot)
dev.off()

# =================================================================================================== -
# >> Perform rfe run 0 ----
# =================================================================================================== -

# set output path
rfe_output_path <- paste0(base_output_path, "/run0")

# candidate set of the number of predictors to evaluate
subsets <- c(seq(from = length(data)-1, to = 10, by = -3),
             seq(from = 9, to = 1, by = -1))

# define all parameters for rfe
response = target_trait
base_learner = "ranger"
type = "regression"
p = 0.8
times = 30
data = data
importance = "permutation"
num.trees = 150
n_cores = 24
parallel = F
savedir = rfe_output_path

# temporary
# candidate set of the number of predictors to evaluate
data <- data %>%
  dplyr::select(diff_area_pp_y_norm_chr, lag_min_p_density, lag_lesion_age_gdd) %>% 
  sample_n(5000)
subsets <- c(seq(from = 2, to = 1, by = -1))
times = 3

# perform rfe
rfe <- perform_rfe(response = response, base_learner = base_learner, type = type,
                   p = p, times = times, 
                   subsets = subsets, data = data,
                   importance = importance,
                   num.trees = num.trees, n_cores = n_cores, parallel = parallel,
                   savedir = savedir)

# =================================================================================================== -
# Evaluate rfe output run 0----
# =================================================================================================== -

# load rfe output
rfe_output_path <- paste0(base_output_path, "/run0/")
subset_output_files <- list.files(rfe_output_path, 
                                  pattern = "[0-9].rds",
                                  full.names = T)

rfe <- lapply(subset_output_files, readRDS)

# create a tidy output
tidy <- tidy_rfe_output(data = rfe, base_learner = "ranger")

# get performance profile
prof_A <- plot_perf_profile(tidy[[1]], metric = "Rsquared")
prof_B <- plot_perf_profile(tidy[[1]], metric = "RMSE")
prof <- ggarrange(prof_A, prof_B, nrow = 1)

# save plot
png(paste0(rfe_output_path, "/perf_prof.png"), width = 10, height = 5, units = 'in', res = 300)
plot(prof)
dev.off()

# get feature ranks
ranks <- tidy[[2]] %>% tibble::rowid_to_column("order") %>% 
  mutate(order = as.numeric(1-order))
ranksplot <- plot_feature_ranks(ranks, top_n = nrow(ranks))
# save plot
png(paste0(rfe_output_path, "/feature_ranks.png"), width = 8, height = 5, units = 'in', res = 300)
plot(ranksplot)
dev.off()

# =================================================================================================== -
# Feature correlations and removal ----
# =================================================================================================== -

top_features <- ranks$var
top_data <- data %>% dplyr::select(any_of(top_features)) %>% 
  filter(complete.cases(.))

# drop factors
top_data <- top_data[, !sapply(top_data, is.factor)]

M<-cor(top_data)

png(paste0(rfe_output_path, "/feature_corr.png"), width = 10, height =10, units = 'in', res = 300)
corrplot(M, addCoef.col = "black", tl.col="black", tl.srt=45,
         method = "color", type="upper", diag=FALSE, 
         number.cex = 0.35, tl.cex= 0.5)
dev.off()

# find highly correlated variables
highly_correlated <- as.data.frame(as.table(M)) %>%
  filter(Var1 != Var2, abs(Freq) > 0.925) %>%
  group_by(Freq) %>% 
  slice(1) %>% ungroup() %>% 
  arrange(desc(abs(Freq))) %>% 
  as.data.frame()

# stop if there are no highly correlated variables left
stopifnot(nrow(highly_correlated) > 0)

# add variable to retain (lower mean rank)
outlines <- list()
for (row in 1:nrow(highly_correlated)){
  line <- highly_correlated[row, ]
  out_line <- highly_correlated[row, ]
  var1 <- as.character(line$Var1)
  var2 <- as.character(line$Var2)
  rank_var1 <- ranks[ranks$var == var1, ]$mean
  rank_var2 <- ranks[ranks$var == var2, ]$mean
  idx = which.max(c(rank_var1, rank_var2))
  out_line$drop <- line[, idx]
  outlines[[row]]<- out_line
}
out_df <- do.call("rbind", outlines)
drop_vars <- unique(out_df$drop)

newdata <- data[, !(names(data) %in% drop_vars)]
saveRDS(newdata, paste0(rfe_output_path, "/reduced_traindat.rds"))

# drop factors
newdata_corr <- newdata[, !sapply(newdata, is.factor)]

M <- cor(newdata_corr)

png(paste0(rfe_output_path, "/feature_corr_postrem.png"), width = 10, height =10, units = 'in', res = 300)
corrplot(M, addCoef.col = "black", tl.col="black", tl.srt=45,
         method = "color", type="upper", diag=FALSE, 
         number.cex = 0.35, tl.cex= 0.5)
dev.off()

# =================================================================================================== -
# >> Perform rfe run 1 on feature subset ----
# =================================================================================================== -

# load filtered data from run1
data <- readRDS(paste0(rfe_output_path, "/reduced_traindat.rds"))

# set new output path
rfe_output_path <- paste0(base_output_path, "/run1")

# candidate set of the number of predictors to evaluate
subsets <- c(seq(from = length(data)-1, to = 10, by = -3),
             seq(from = 9, to = 1, by = -1))

# perform rfe
savedir = rfe_output_path
rfe <- perform_rfe(response = response, base_learner = base_learner, type = type,
                   p = p, times = times, 
                   subsets = subsets, data = data,
                   importance = importance,
                   num.trees = num.trees, n_cores = n_cores, parallel = parallel,
                   savedir = savedir)

# =================================================================================================== -
# Evaluate rfe output run 1 ----
# =================================================================================================== -

# load rfe output
subset_output_files <- list.files(rfe_output_path, 
                                  pattern = "[0-9].rds",
                                  full.names = T)

rfe <- lapply(subset_output_files, readRDS)

# create a tidy output
tidy <- tidy_rfe_output(rfe, base_learner = "ranger")

# get performance profile
prof_A <- plot_perf_profile(tidy[[1]], metric = "Rsquared")
prof_B <- plot_perf_profile(tidy[[1]], metric = "RMSE")
prof <- ggarrange(prof_A, prof_B, nrow = 1)

# save plot
png(paste0(rfe_output_path, "/perf_prof.png"), width = 10, height = 6, units = 'in', res = 300)
plot(prof)
dev.off()

# get feature ranks
ranks <- tidy[[2]] %>% tibble::rowid_to_column("order") %>% 
  mutate(order = as.numeric(1-order))
ranksplot <- plot_feature_ranks(ranks, top_n = nrow(ranks))
# save plot
png(paste0(rfe_output_path, "/feature_ranks.png"), width = 8, height = 4, units = 'in', res = 300)
plot(ranksplot)
dev.off()

# =================================================================================================== -
# Feature correlations and removal ----
# =================================================================================================== -

top_features <- ranks$var
top_data <- data %>% dplyr::select(any_of(top_features)) %>% 
  filter(complete.cases(.))

# drop factors
top_data <- top_data[, !sapply(top_data, is.factor)]

M<-cor(top_data)

png(paste0(rfe_output_path, "/feature_corr.png"), width = 10, height =10, units = 'in', res = 300)
corrplot(M, addCoef.col = "black", tl.col="black", tl.srt=45,
         method = "color", type="upper", diag=FALSE, 
         number.cex = 0.35, tl.cex= 0.5)
dev.off()

# find highly correlated variables
highly_correlated <- as.data.frame(as.table(M)) %>%
  filter(Var1 != Var2, abs(Freq) > 0.85) %>%
  group_by(Freq) %>% 
  slice(1) %>% ungroup() %>% 
  arrange(desc(abs(Freq))) %>% 
  as.data.frame()

# stop if there are no highly correlated variables left
stopifnot(nrow(highly_correlated) > 0)

# add variable to retain (lower mean rank)
ranks
outlines <- list()
for (row in 1:nrow(highly_correlated)){
  line <- highly_correlated[row, ]
  out_line <- highly_correlated[row, ]
  var1 <- as.character(line$Var1)
  var2 <- as.character(line$Var2)
  rank_var1 <- ranks[ranks$var == var1, ]$mean
  rank_var2 <- ranks[ranks$var == var2, ]$mean
  idx = which.max(c(rank_var1, rank_var2))
  out_line$drop <- line[, idx]
  outlines[[row]]<- out_line
}
out_df <- do.call("rbind", outlines)
drop_vars <- unique(out_df$drop)

newdata <- data[, !(names(data) %in% drop_vars)]
saveRDS(newdata, paste0(rfe_output_path, "/reduced_traindat.rds"))

# drop factors
newdata_corr <- newdata[, !sapply(newdata, is.factor)]

M <- cor(newdata_corr)

png(paste0(rfe_output_path, "/feature_corr_postrem.png"), width = 10, height =10, units = 'in', res = 300)
corrplot(M, addCoef.col = "black", tl.col="black", tl.srt=45,
         method = "color", type="upper", diag=FALSE, 
         number.cex = 0.35, tl.cex= 0.5)
dev.off()

# =================================================================================================== -
# >> Perform rfe run 2 on feature subset ----
# =================================================================================================== -

# load filtered data from run1
data <- readRDS(paste0(rfe_output_path, "/reduced_traindat.rds"))

# set new output path
rfe_output_path <- paste0(base_output_path, "/run2")

# candidate set of the number of predictors to evaluate
subsets <- c(seq(from = length(data)-1, to = 10, by = -3),
             seq(from = 9, to = 1, by = -1))

# perform rfe
savedir = rfe_output_path
rfe <- perform_rfe(response = response, base_learner = base_learner, type = type,
                   p = p, times = times, 
                   subsets = subsets, data = data,
                   importance = importance,
                   num.trees = num.trees, n_cores = n_cores, parallel = parallel,
                   savedir = savedir)

# =================================================================================================== -
# Evaluate rfe output run 2 ----
# =================================================================================================== -

# load rfe output
subset_output_files <- list.files(rfe_output_path, 
                                  pattern = "[0-9].rds",
                                  full.names = T)

rfe <- lapply(subset_output_files, readRDS)

# create a tidy output
tidy <- tidy_rfe_output(rfe, base_learner = "ranger")

# get performance profile
prof_A <- plot_perf_profile(tidy[[1]], metric = "Rsquared")
prof_B <- plot_perf_profile(tidy[[1]], metric = "RMSE")
prof <- ggarrange(prof_A, prof_B, nrow = 1)

# save plot
png(paste0(rfe_output_path, "/perf_prof.png"), width = 10, height = 6, units = 'in', res = 300)
plot(prof)
dev.off()

# get feature ranks
ranks <- tidy[[2]] %>% tibble::rowid_to_column("order") %>% 
  mutate(order = as.numeric(1-order))
ranksplot <- plot_feature_ranks(ranks, top_n = nrow(ranks))
# save plot
png(paste0(rfe_output_path, "/feature_ranks.png"), width = 8, height = 4, units = 'in', res = 300)
plot(ranksplot)
dev.off()

# # =================================================================================================== -
# # =================================================================================================== -
# 
# # END WORK START FUN
# 
# # =================================================================================================== -
# # >> sequential subset plotting ----
# # =================================================================================================== - 
# 
# library(viridis)
# 
# # single rf model fit
# train <- data
# mtry <- ceiling(seq(ceiling(0.1*length(train[-1])), ceiling(0.66*length(train[-1])), len = 6)) %>% unique()
# min.node.size <- c(5)
# num.trees = 150
# response = target_trait
# base_learner = "ranger"
# tune_grid <- expand.grid(mtry = mtry,
#                          splitrule = "variance",
#                          min.node.size = min.node.size) 
# 
# #define inner resampling procedure
# ctrl <- caret::trainControl(method = "repeatedcv",
#                             number = 7,
#                             rep = 1,
#                             verbose = TRUE,
#                             allowParallel = TRUE,
#                             savePredictions = TRUE,
#                             classProbs = FALSE)
# 
# #define model to fit
# formula <- as.formula(paste(response, " ~ .", sep = ""))
# 
# #tune/train random forest
# fit <- caret::train(formula,
#                     data = train,
#                     preProc = c("center", "scale"),
#                     method = base_learner,
#                     tuneGrid = tune_grid,
#                     trControl = ctrl,
#                     num.trees = 150,
#                     importance = "permutation")
# 
# saveRDS(fit, paste0(base_output_path, "/rf_fit.rds"))
# 
# # =================================================================================================== - 
# 
# fit <- readRDS(paste0(base_output_path, "/rf_fit.rds"))
# imp <- varImp(fit)$importance %>% 
#   rownames_to_column("var") %>% as_tibble()
# 
# # Sum up importance for factor levels
# if(any(sapply(train, is.factor))){
#   fct_chr_cols <- train %>%
#     select(where(~ is.factor(.) || is.character(.))) %>%
#     colnames()
#   # remove single factor level importance values
#   drop_fct_imp <- paste0("^", fct_chr_cols, collapse = "|")
#   drop_fct_imp <- paste(imp$var[grep(drop_fct_imp, imp$var)], collapse = "|")
#   # Aggregate importance for factor variables
#   fct_agg_imp <- imp %>%
#     mutate(
#       var = map_chr(imp$var, function(var) {
#         # Match variable name with factor column prefixes
#         match <- fct_chr_cols[which(startsWith(var, fct_chr_cols))]
#         if (length(match) > 0) match else NA_character_
#       })
#     ) %>%
#     group_by(var) %>%
#     summarize(
#       Overall = sum(Overall, na.rm = TRUE)          ) %>%
#     filter(!is.na(var))  # Remove non-factor variables
#   # combine factor importance with numeric variable importance
#   imp <- imp[!grepl(drop_fct_imp, imp$var),]
#   imp <- bind_rows(imp, fct_agg_imp)
# }
# 
# # imitate the ranks table based on the single run
# imp <- imp %>% arrange(desc(Overall))
# imp$order <- c(0:-46)
# imp$mean <- 1:47
# imp$sd <- rep(NA, 47)
# imp$se <- rep(NA, 47)
# imp <- imp %>% dplyr::select(order, var, mean, sd, se)
# 
# # assign each variable a specific color
# color_list <- viridis_pal()(nrow(imp))
# label_colors <- c()
# for (j in seq_along(imp$var)) {
#   label_colors[imp$var[j]] <- color_list[j]
# }
# 
# # =================================================================================================== - 
# 
# # First, make the basic plot based on the single rf run
# 
# rfe_output_path <- paste0(base_output_path, "/run0")
# output_path <- paste0(rfe_output_path, "/sequential")
# 
# plot_feature_ranks <- function(data, top_n, label_colors) {
#   
#   data <- data %>% slice(1:top_n)
#   
#   ggplot(data, aes(x = order, y = mean)) +
#     geom_point() +
#     geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.5) +
#     ylab("Feature rank") +
#     xlab("Feature") +
#     # Add categories to axis
#     scale_x_continuous(
#       breaks = data$order,
#       labels = data$var,
#       expand = c(0, 0)
#     ) +
#     coord_flip() +
#     theme_bw() %+replace%
#     theme(
#       axis.title.y = element_blank(),
#       plot.title = element_text(size = 15, face = "bold"),
#       panel.grid.minor = element_blank(),
#       panel.grid.major.x = element_blank(),
#       # Dynamically assign colors to y-axis text
#       axis.text.y = element_text(face = "bold"),
#       axis.text.y.left = element_text(color = label_colors[data$var])
#     )
# }
# 
# # null plot (single rf run)
# ranksplot <- plot_feature_ranks(data=imp, top_n = nrow(imp), label_colors=label_colors)
# png(paste0(output_path, "/feature_ranks_0.png"), width = 8, height = 6, units = 'in', res = 300)
# plot(ranksplot)
# dev.off()
# 
# # Then make all plots including increasingly more RFE runs
# # load rfe output
# subset_output_files <- list.files(rfe_output_path, 
#                                   pattern = ".rds",
#                                   full.names = T)
# 
# for (i in 1:length(subset_output_files)) {
#   
#   rfe <- lapply(subset_output_files[1:i], readRDS)
#   
#   # create a tidy output
#   tidy <- tidy_rfe_output(rfe, base_learner = "ranger")
#   
#   # get performance profile
#   prof <- plot_perf_profile(tidy[[1]])
#   prof <- prof +
#     scale_y_continuous(limits = c(4, 6.5))
#   # save plot
#   png(paste0(output_path, "/perf_prof_", i, ".png"), width = 5, height = 3, units = 'in', res = 300)
#   plot(prof)
#   dev.off()
#   
#   # get feature ranks
#   ranks <- tidy[[2]] %>% tibble::rowid_to_column("order") %>% 
#     mutate(order = as.numeric(1-order))
#   ranksplot <- plot_feature_ranks(ranks, top_n = nrow(ranks), label_colors=label_colors)
#   # save plot
#   png(paste0(output_path, "/feature_ranks_", i, ".png"), width = 8, height = 6, units = 'in', res = 300)
#   plot(ranksplot)
#   dev.off()
# }
# 
# # =================================================================================================== - 
# # predicted vs observed for the final model ----
# # =================================================================================================== - 
# 
# library(ggpubr)
# library(ggpmisc)
# 
# # load rfe output
# rfe_output_path <- paste0(base_output_path, "/run2")
# subset_output_files <- list.files(rfe_output_path, 
#                                   pattern = "[0-9].rds",
#                                   full.names = T)
# rfe <- lapply(subset_output_files, readRDS)
# 
# # create a tidy output
# tidy <- tidy_rfe_output(rfe, base_learner = "ranger")
# 
# # get feature ranks
# ranks <- tidy[[2]] %>% tibble::rowid_to_column("order") %>% 
#   mutate(order = as.numeric(1-order))
# 
# # forward
# R <- list()
# for (j in 1:12){
#   r <- vector()
#   for (i in 1:nrow(ranks)){
#   # for (i in 1:5){
#     top_n <- paste0(ranks$var[1:i], collapse = "|")
#     top_n <- paste(target_trait, top_n, sep = "|")
#     
#     # single rf model fit
#     train <- data[, grepl(top_n, colnames(data))]
#     mtry <- ceiling(seq(ceiling(0.1*length(train[-1])), ceiling(0.66*length(train[-1])), len = 6)) %>% unique()
#     min.node.size <- c(5)
#     num.trees = 150
#     response = target_trait
#     base_learner = "ranger"
#     tune_grid <- expand.grid(mtry = mtry,
#                              splitrule = "variance",
#                              min.node.size = min.node.size) 
#     
#     #define inner resampling procedure
#     ctrl <- caret::trainControl(method = "repeatedcv",
#                                 number = 7,
#                                 rep = 1,
#                                 verbose = TRUE,
#                                 allowParallel = TRUE,
#                                 savePredictions = TRUE,
#                                 classProbs = FALSE)
#     
#     #define model to fit
#     formula <- as.formula(paste(response, " ~ .", sep = ""))
#     
#     #tune/train random forest
#     fit <- caret::train(formula,
#                         data = train,
#                         preProc = c("center", "scale"),
#                         method = base_learner,
#                         tuneGrid = tune_grid,
#                         trControl = ctrl,
#                         num.trees = 150,
#                         importance = "permutation")
#     
#     predobs_cv <- plyr::match_df(fit$pred, fit$bestTune, on = names(fit$bestTune))
#     #Average predictions of the held out samples;
#     predobs <- predobs_cv %>% 
#       group_by(rowIndex) %>% 
#       dplyr::summarize(obs = mean(obs),
#                        mean_pred = mean(pred))
#     
#     r[i] <- cor(predobs$obs, predobs$mean_pred)
#     
#     # Pairwise correlations
#     p <- ggplot(data = predobs, aes(x = mean_pred, y = obs)) +
#       geom_abline(intercept = 0, slope = 1) +
#       geom_smooth(method = "lm") +
#       geom_point(size = 2) +
#       # stat_correlation(aes(label = after_stat(cor)), size = 3, geom = "label_npc", alpha = 0.1) +
#       stat_correlation(mapping = use_label(c("R", "P", "n")), size = 3, geom = "label_npc") +
#       xlab("predicted") + ylab("observed") +
#       scale_x_continuous(limits = c(-0.0025, 0.0075)) +
#       theme(panel.grid = element_line(linewidth = 0.2),
#             strip.background = element_blank(),
#             panel.background = element_rect(fill = "#ebedf0"))
#     png(paste0(rfe_output_path, "/predobs/predobs_top_", i, ".png"), width = 5, height = 3, units = 'in', res = 300)
#     plot(p)
#     dev.off()
#     
#   }
#   
#   # collect all r
#   R[[j]] <- r
#   
# }
# 
# saveRDS(R, paste0(rfe_output_path, "/predobs/R.rds"))
# 
# # Convert list to tibble
# result <- as_tibble(setNames(R, paste0("V", seq_along(R))))
# 
# # Compute mean and SE per row
# df_summary <- result %>%
#   rowwise() %>%
#   mutate(
#     mean = mean(as.numeric(c_across(everything())), na.rm = TRUE),
#     SD = sd(as.numeric(c_across(everything())), na.rm = TRUE) 
#   ) %>% 
#   ungroup() %>% 
#   rownames_to_column() %>% 
#   mutate(rowname= as.numeric(rowname))
# 
# pd <- position_dodge(0.5) # move them .05 to the left and right
# ggplot(df_summary, aes(x = rowname, y = mean)) +
#   geom_point() +
#   geom_line() +
#   geom_errorbar(position = pd, aes(ymin = mean - SD, ymax = mean + SD), width = 1, alpha = 0.5) +
#   scale_y_continuous(limits = c(0, 0.5))
# 
# 
# 
# 

# Skip to line 57 for the simpler example.
# Skip to line 197 for the more complex example (hyperparameter tuning & more robust model evaluation).


# ========================= #
# == Prepare environment == #
# ========================= #

# install.packages(c('tidyverse', 'tidymodels', 'kknn', 'xgboost', 'ranger', 'palmerpenguins'))

library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(kknn)
library(xgboost)
library(ranger)


# Clear the environment
remove(list=ls())

# ========================== #
# == Exploratory Analysis == #
# ========================== #

# Inspect data
summary(penguins)
penguins %>% head

# Visualize relationships
penguins %>% 
  ggplot(aes(flipper_length_mm, bill_length_mm, color = sex))+
  geom_point()

penguins %>% 
  filter(!is.na(sex)) %>% 
  ggplot(aes(flipper_length_mm, bill_length_mm, color = sex))+
  geom_point()

penguins %>% 
  filter(!is.na(sex)) %>% 
  ggplot(aes(flipper_length_mm, bill_length_mm, color = species))+
  geom_point()

penguins %>% 
  filter(!is.na(sex)) %>% 
  ggplot(aes(flipper_length_mm, bill_length_mm, color = sex))+
  geom_point()+
  facet_wrap(~species)

# Feature selection: choose the variables we want to include
penguin_df <- penguins %>% 
  filter(!is.na(sex)) %>% 
  select(-year, -island)


# ============================================= #
# == Simple Model (No Hyperparameter Tuning) == #
# ============================================= #

# Set the random seed for reproducibility
set.seed(123)

# Split the observations into train & test sets
penguin_split  <- initial_split(penguin_df, prop=0.75)
penguin_train  <- training(penguin_split)
penguin_test   <- testing(penguin_split)

# At this point, we should have 2 datasets: train and test
nrow(penguin_train)
nrow(penguin_test)
nrow(penguin_df)
rbind(penguin_train, penguin_test) %>% duplicated %>% sum     # should be 0

# Store the model specifications, just using default hyper-parameter values
rf_spec <- 
  rand_forest() %>%                     # choose our ML algorithm
  set_engine("ranger") %>%              # choose a specific R package
  set_mode("classification")            # either classification or regression

knn_spec <- 
  nearest_neighbor(neighbors = 4) %>%   
  set_engine("kknn") %>%                
  set_mode("classification")            

xgb_spec <- 
  boost_tree() %>%                      
  set_engine("xgboost") %>%             
  set_mode("classification")            

logistic_spec <- 
  logistic_reg() %>%                    
  set_engine("glm")                     # logistic regression is a different beast

# Define the recipes (formula & preprocessing steps)
penguins_recipe <- 
  recipe(sex ~ species + bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, 
         data = penguin_train) %>%
  step_normalize(all_numeric()) %>%             # this isn't really necessary for RFs
  step_impute_knn(all_predictors()) %>%         # again, not needed for RFs
  step_dummy(all_nominal(), -all_outcomes())    # likewise here, only needed for the other models

# Set a simple workflow for each ML model
rf_workflow <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(rf_spec)

knn_workflow <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(knn_spec)

xgb_workflow <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(xgb_spec)

logistic_workflow <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(logistic_spec)

# Train the models on the training data
set.seed(234)
rf_model       <- fit(rf_workflow, penguin_train)
knn_model      <- fit(knn_workflow, penguin_train)
xgb_model      <- fit(xgb_workflow, penguin_train)
logistic_model <- fit(logistic_workflow, penguin_train)

# Make predictions on test set
rf_predict       <- predict(rf_model, penguin_test)
knn_predict      <- predict(knn_model, penguin_test)
xgb_predict      <- predict(xgb_model, penguin_test)
logistic_predict <- predict(logistic_model, penguin_test)

# Get an unbiased estimate of model accuracy by evaluating the test set predictions
bind_cols(rf_predict, penguin_test) %>% 
  metrics(truth = sex, estimate = .pred_class)

bind_cols(knn_predict, penguin_test) %>% 
  metrics(truth = sex, estimate = .pred_class)

bind_cols(xgb_predict, penguin_test) %>% 
  metrics(truth = sex, estimate = .pred_class)

bind_cols(logistic_predict, penguin_test) %>% 
  metrics(truth = sex, estimate = .pred_class)

# ====================================== #
# == Overview of Tidymodels Functions == #
# ====================================== #

# Functions we saw in the simple model
data_split <- initial_split(full_dataset)
training_df <- training(data_split)
testing_df <- testing(data_split)

model_specification <- 
  rand_forest() %>%                     # choose our ML algorithm
  set_engine("ranger") %>%              # choose a specific R package
  set_mode("classification")            # either classification or regression

model_recipe <- recipe(y ~ x formula, data = dataframe_template)

wf_object <- workflow() %>% 
  add_recipe(model_recipe) %>% 
  add_model(model_specification)

fitted_model <- fit(wf_object, training_data)
predctions <- predict(fitted_model, test_data)
metrics(truth = actual_y_variable, estimate = predicted_y_variable)


# New functions used in the complex model
vfold_cv(training_df)                                                    # create V resamples/folds that cycle through the different 
# holdout assessment sets

expand.grid(list_of_hyperparameters_to_tune_and_their_possible_values)   # create a hyperparameter search space

tune_results <- tune_grid(...)                                           # perform cross validation and apply the train/test process 
# to the CV resamples

finalize_workflow(old_workflow, final_tuned_hyperparameters)             # feed the tuned hyperparameters into the previous workflow

collect_metrics(tune_results)                                            # extract accuracy, AUC, and other metrics we specified

select_best(tune_results)                                                # extract the best model from cross-validation

collect_predictions(tune_results)                                        # extract the predictions on the holdout set from the CV process

conf_mat(predictions_object)                                             # create a confusion matrix 

conf_mat_resampled(tune_results)                                         # for a given model, get an avg. confusion matrix showing its performance
# performance across all V resamples, cycling thru holdout assessment sets


# =========================================== #
# == Complex Model (Hyperparameter Tuning) == #
# =========================================== #

# Load the packages from last time
library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(kknn)
library(xgboost)
library(ranger)

# Add a couple additional packages
# install.packages(c('vip','gridExtra'))
library(vip)        # for extracting variable importance
library(gridExtra)  # for visualizing multiple ggplots at once


# 1. Feature selection: choose the variables we want to include, 
#    and remove observations with missing outcome variable
penguin_df <- penguins %>% 
  filter(!is.na(sex)) %>% 
  select(-year, -island)

# 2. Split the observations into train & test sets
set.seed(123)     # Set the random seed for reproducibility
penguin_split  <- initial_split(penguin_df, prop=0.75)
penguin_train  <- training(penguin_split)
penguin_test   <- testing(penguin_split)

# 3. Create a cross-validation set from the training data
set.seed(1357)
penguin_cv <- vfold_cv(penguin_train, v=10, repeats=1)

# If there's class imbalance (eg, far more females than males), stratifying by the outcome variable can be important
penguin_train$sex %>% table
penguin_cv <- vfold_cv(penguin_train, strata=sex, v=10, repeats=1)

# To see the 1st fold:
penguin_cv$splits[1] %>% data.frame

# 4. Set a workflow (specify the model, define a recipe, and combine)
rf_spec_with_tuning <- 
  rand_forest(
    trees=tune(),
    mtry=tune(),
    min_n=tune()) %>% # min num of observations that have to be left to consider another split
  set_engine("ranger", 
             importance="impurity") %>% 
  set_mode("classification") 

penguins_recipe <- 
  recipe(sex ~ species + bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, 
         data = penguin_train)
#step_normalize() %>%
#step_impute_knn(all_predictors()) %>%
#step_dummy(all_nominal(), -all_outcomes())

rf_workflow.v2 <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(rf_spec_with_tuning)

# 5. Store the hyperparameter search space (a search "grid")
expand.grid(1:3, LETTERS[1:3]) # gives every combo of 1,2,3 and a,b,c (just an example of what expand.grid does)

rf_grid <- expand.grid(trees = c(500),
                       mtry  = c(3, 4, 5),
                       min_n = c(5, 10))

# 6. Find the tuned hyperparameters using cross-validation 
set.seed(123456)
rf_tune_results <- 
  tune_grid(object    = rf_workflow.v2,
            resamples = penguin_cv,
            grid      = rf_grid,
            metrics   = metric_set(accuracy, roc_auc),
            control   = control_grid(save_pred = TRUE))  #this controls aspects of the grid search

# 7. Identify the best model from cross-validation
collect_metrics(rf_tune_results) %>% View
select_best(rf_tune_results, metric = "accuracy")

# 8. Look for patterns in the misclassified examples (for the best model)
collect_predictions(rf_tune_results) %>% View

collect_predictions(rf_tune_results) %>% 
  filter(sex!=.pred_class) %>%                # only look at examples that were misclassified
  filter(mtry==5, min_n==5) %>%               # this was the best model
  View

bestMod_CVpreds <- 
  collect_predictions(rf_tune_results) %>% 
  filter(mtry==5, min_n==5) %>%                          # this was the best model
  select(.row, .pred_class, .pred_female, .pred_male)    # select columns we want to merge with the original dataset

bestMod_CVpreds <- bind_cols(bestMod_CVpreds, penguin_train[bestMod_CVpreds$.row,])

# We're misclassifying more Adelie penguins than Chinstrap or Gentoo
num_misclassified_by_species <-
  bestMod_CVpreds %>% 
  filter(.pred_class!=sex) %>% 
  group_by(species) %>% 
  summarise(n_misclassified = n()) %>% 
  select(n_misclassified) %>% 
  unlist

num_of_each_species <- table(penguin_train$species)

num_misclassified_by_species/num_of_each_species

# 9. Inspect confusion matrix on CV set for best model
conf_mat(data=bestMod_CVpreds,
         truth=sex, 
         estimate=.pred_class)

resampled_conf_matrix_from_CV <- 
  conf_mat_resampled(
    x=rf_tune_results, 
    parameters=select_best(rf_tune_results, metric = "accuracy")
  )

resampled_conf_matrix_from_CV %>% 
  pivot_wider(names_from='Prediction', values_from='Freq')

# 10. Make plots to look for patterns in misclassified examples

# m vs FL
a <- bestMod_CVpreds %>% 
  filter(species=="Adelie") %>% 
  ggplot(aes(x=body_mass_g, y=flipper_length_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

# m vs BD
b <- bestMod_CVpreds %>%
  filter(species=="Adelie") %>% 
  ggplot(aes(x=body_mass_g, y=bill_depth_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

# m vs BL
c <- bestMod_CVpreds %>% 
  filter(species=="Adelie") %>% 
  ggplot(aes(x=body_mass_g, y=bill_length_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

# FL vs BD
d <- bestMod_CVpreds %>%
  filter(species=="Adelie") %>% 
  ggplot(aes(x=bill_depth_mm, y=flipper_length_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

# FL vs BL
e <- bestMod_CVpreds %>%
  filter(species=="Adelie") %>% 
  ggplot(aes(x=bill_length_mm, y=flipper_length_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

# BD vs BL
f <- bestMod_CVpreds %>%
  filter(species=="Adelie") %>% 
  ggplot(aes(x=bill_depth_mm, y=bill_length_mm))+
  geom_point(pch=21, aes(fill=sex, color=.pred_class), size=2, stroke = 1.5)

a
b
c
d
e
f
grid.arrange(a,b,c,d,e,f, ncol=2)

# 11. Check learning curves (NOTE: before running this code, store the functions at line 422)
rf_spec_for_LC <- 
  rand_forest(mtry=3, trees=500, min_n=5) %>%         # these were the best, tuned hyperparameters 
  set_engine("ranger", importance="impurity") %>% 
  set_mode("classification")

penguins_recipe <- 
  recipe(sex ~ species + bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g, 
         data = penguin_train)

rf_workflow.LC <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(rf_spec_for_LC)

set.seed(2468)
LC_data <- learning_curves_data(penguin_cv, 20, 10, penguin_df, rf_workflow.LC)
plot_learning_curves(LC_data)


# 12. Make additional improvements (feature engineering, new model specification, 
#     different hyperparameters, etc.)
# .
# .
# .

# 13. Finalize the workflow and extract the best hyperparameter values
param_final <- select_best(rf_tune_results, metric = "accuracy")

# 14. Feed them into a final workflow
rf_workflow.v2 <-  finalize_workflow(rf_workflow.v2, param_final)

# 15. Train the final model
set.seed(54321)
final_rf_fit <- fit(rf_workflow.v2, penguin_train)

# 16. Inspect variable importance of the final model trained on all of the training data
final_rf_fit %>% 
  extract_fit_parsnip %>% 
  vip

# 17. Store the test predictions
rf_test_preds2 <- predict(final_rf_fit, penguin_test)

# 18. Evaluate the model's accuracy on the test set
bind_cols(rf_test_preds2, penguin_test) %>% 
  metrics(truth = sex, estimate = .pred_class)

bind_cols(rf_test_preds2, penguin_test) %>% 
  conf_mat(truth = sex, estimate = .pred_class)









###################################
## Functions for learning curves ##
###################################

# These functions are adapted from https://github.com/tidymodels/rsample/issues/166

remove_random <- function(split, prop) {
  if (prop >= 1) {
    return(split$in_id)
  }
  l <- length(split$in_id)
  p <- round(l * (1 - prop))
  split$in_id[-sample(1:l, p)]
}

learning_curves_data <- function(cv_folds, num_breaks, num_folds, training_df, wf){
  cv_folds %>%
    crossing(prop = c(seq(1/num_breaks, 1, by=1/num_breaks))) %>%
    mutate(analysis = purrr::map2(splits, prop, remove_random),
           assessment = purrr::map(splits, complement),
           splits = purrr::map2(analysis, 
                                assessment, 
                                ~make_splits(list(analysis = .x, assessment = .y), 
                                             training_df))) %>%
    select(prop, splits) %>%
    nest(learning_splits = c(splits)) %>%
    mutate(learning_splits = purrr::map(learning_splits, manual_rset, paste0("LearningFold", 1:num_folds))) %>% 
    mutate(res = map(learning_splits, ~fit_resamples(wf, .)),
           metrics = map(res, collect_metrics)) %>%
    unnest(metrics)
}

plot_learning_curves <- function(learning_curve_dataset){
  learning_curve_dataset %>%
    ggplot(aes(prop, mean, color = .metric)) +
    geom_ribbon(aes(ymin = mean - std_err,
                    ymax = mean + std_err), alpha = 0.3, color = NA) +
    geom_line(alpha = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~.metric, ncol = 1, scales = "free_y") +
    theme(legend.position = "none") +
    labs(y = NULL)
}



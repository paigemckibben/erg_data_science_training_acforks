### Presentation Key####

# ========================= #
# == Prepare environment == #
# ========================= #

install.packages(c('tidyverse', 'tidymodels', 'kknn', 'xgboost', 'ranger', 'palmerpenguins'))

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
set.seed(1234)

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
         data = penguin_df) %>%
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
set.seed(2345)
rf_model       <- fit(rf_workflow, penguin_train)
knn_model      <- fit(knn_workflow, penguin_train)
xgb_model      <- fit(xgb_workflow, penguin_train)
logistic_model <- fit(logistic_workflow, penguin_train)

# Make predictions on test set
rf_predict <- predict(rf_model, penguin_test)
knn_predict <- predict(knn_model, penguin_test)
xgb_predict <- predict(xgb_model, penguin_test)
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



# =========================================== #
# == Complex Model (Hyperparameter Tuning) == #
# =========================================== #


# 1. Create a cross-validation set from the training data
penguin_cv <- vfold_cv(penguin_train)

# 2. Store the model specification
rf_spec_with_tuning <- 
  rand_forest() %>%
  set_args(trees=tune(),
           mtry=tune(),
           min_n=tune()) %>% 
  set_engine("ranger", 
             importance="impurity") %>% 
  set_mode("classification") 

# 3. Set a workflow (using the previous recipe)
rf_workflow.v2 <- workflow() %>% 
  add_recipe(penguins_recipe) %>% 
  add_model(rf_spec_with_tuning)

# 4. Store the hyperparameter search space (a search "grid")
rf_grid <- expand.grid(mtry  = c(2, 4),
                       trees = c(100, 500),
                       min_n = c(10))

# 5. Find the tuned hyperparameters using cross-validation 
rf_tune_results <- rf_workflow.v2 %>% 
  tune_grid(resamples = penguin_cv,
            grid = rf_grid,
            metrics = metric_set(accuracy, roc_auc))

rf_tune_results %>% collect_metrics() %>% View

# 6. Finalize the workflow and extract the best hyperparameter values
param_final <- rf_tune_results %>% 
  select_best(metric = "accuracy")

# 7. Feed them into a final workflow
rf_workflow.v2 <- rf_workflow.v2 %>% 
  finalize_workflow(param_final)

# 8. Train the final model
rf_fit <- fit(rf_workflow.v2, penguin_train)

# 9. Evaluate the confusion matrix
training_preds <- predict(rf_fit, penguin_train)

bind_cols(training_preds, penguin_train) %>% 
  conf_mat(truth = sex, estimate = .pred_class)

# 10. Look for patterns in the incorrect predictions
cbind(penguin_train, training_preds) %>% 
  filter(sex!=.pred_class)

# 11. Store the test predictions
rf_test_preds2 <- predict(rf_fit, penguin_test)

# 12. Evaluate the model's accuracy on the test set
bind_cols(penguin_test, rf_test_preds2) %>% 
  metrics(truth = sex, estimate = .pred_class)










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
summary()
head()

# Visualize relationships
ggplot()+
  geom_point()



# Feature selection: choose the variables we want to include



# ============================================= #
# == Simple Model (No Hyperparameter Tuning) == #
# ============================================= #

# Set the random seed for reproducibility
set.seed()

# Split the observations into train & test sets
initial_split()
training()
testing()

# At this point, we should have 2 datasets: train and test
nrow()
nrow()
nrow()


# Store the model specifications, just using default hyper-parameter values

  rand_forest() %>%
  set_engine() %>%
  set_mode()

  nearest_neighbor() %>%   
  set_engine() %>%                
  set_mode()            

  boost_tree() %>%                      
  set_engine() %>%             
  set_mode()            

  logistic_reg() %>%                    
  set_engine()

# Define the recipes (formula & preprocessing steps)
  recipe() %>%
  step_normalize() %>%
  step_impute_knn() %>%
  step_dummy()

# Set a simple workflow for each ML model

  workflow() %>% 
  add_recipe() %>% 
  add_model()


# Train the models on the training data
set.seed()
fit()
fit()
fit()
fit()

# Make predictions on test set
predict()
predict()
predict()
predict()

# Get an unbiased estimate of model accuracy by evaluating the test set predictions
metrics(truth = , 
        estimate = )

metrics(truth = , 
        estimate = )

metrics(truth = , 
        estimate = )

metrics(truth = , 
        estimate = )




# =========================================== #
# == Complex Model (Hyperparameter Tuning) == #
# =========================================== #


# 1. Create a cross-validation set from the training data
vfold_cv()

# 2. Store the model specification

  rand_forest() %>%
  set_args(trees= ,
           mtry= ,
           min_n= ) %>% 
  set_engine() %>% 
  set_mode() 

# 3. Set a workflow (using the previous recipe)
rf_workflow.v2 <- workflow() %>% 
  add_recipe() %>% 
  add_model()

# 4. Store the hyperparameter search space (a search "grid")
rf_grid <- expand.grid(mtry  = ,
                       trees = ,
                       min_n = )

# 5. Find the tuned hyperparameters using cross-validation 

  tune_grid(resamples = ,
            grid = ,
            metrics = )

collect_metrics()

# 6. Finalize the workflow and extract the best hyperparameter values
select_best()

# 7. Feed them into a final workflow
finalize_workflow()

# 8. Train the final model
fit()

# 9. Evaluate the confusion matrix
predict()
conf_mat()

# 10. Look for patterns in the incorrect predictions


# 11. Store the test predictions
predict()

# 12. Evaluate the model's accuracy on the test set
metrics(truth = , 
        estimate = )




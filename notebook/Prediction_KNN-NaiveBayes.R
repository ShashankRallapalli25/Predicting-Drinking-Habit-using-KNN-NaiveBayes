#Group project R Code - Team Delta

# Load necessary libraries and install them if required
#install.packages("tidyverse")
#install.packages("caret")
#install.packages("ggcorrplot")
#install.packages("e1071")
#install.packages("randomForest")
#install.packages("class")
#install.packages("glmnet")
library(tidyverse)  # For data manipulation and visualization
library(caret)      # For machine learning
library(ggcorrplot)
library(e1071)
library(randomForest)
library(class)
library(glmnet)

# Load the data set from a file dialog. Download the submitted data set (smoking_driking_dataset_Ver01)
dataset <- read.csv(file.choose())

# Display the dimensions of the dataset
shape <- dim(dataset)
cat("Number of Rows:", shape[1], "\n")
cat("Number of Columns:", shape[2], "\n")

# Display the first few rows of the data set
head(dataset)

# Display data set information, e.g., column names, data types, and non-null counts
str(dataset)

# EDA: Creating EDA Plots (at least five)
# Bar plot for alcohol consumption by gender
dataset %>%
  ggplot(aes(x = sex, fill = DRK_YN)) +
  geom_bar(position = "fill") +
  labs(title = "Alcohol Consumption by Gender")

# Histogram for age distribution
dataset %>%
  ggplot(aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Age Distribution")

# Box plot for alcohol consumption by age and gender
dataset %>%
  ggplot(aes(x = sex, y = age, fill = DRK_YN)) +
  geom_boxplot() +
  labs(title = "Alcohol Consumption by Age and Gender")

# Scatter Plot of HDL Cholesterol vs. Triglycerides
ggplot(dataset, aes(x = HDL_chole, y = triglyceride)) +
  geom_point() +
  labs(title = "HDL Cholesterol vs. Triglycerides", x = "HDL Cholesterol", y = "Triglycerides")

# Box plot of weight by drinking habit
ggplot(dataset, aes(x = DRK_YN, y = weight, fill = DRK_YN)) +
  geom_boxplot() +
  labs(title = "Weight by Drinking Habit", x = "Drinking Habit", y = "Weight") +
  scale_fill_manual(values = c("N" = "blue", "Y" = "red"))

# Data preprocessing encoding the variables Sex and target variable as 0 and 1
dataset$sex <- ifelse(dataset$sex == "Male", 0, 1)
dataset$DRK_YN <- ifelse(dataset$DRK_YN == "Y", 1, 0)

# Calculate and visualize the correlation matrix
correlation_matrix <- cor(dataset[, c('DRK_YN','sex','age', 'height', 'weight', 'SBP', 'DBP', 'HDL_chole', 'LDL_chole', 'triglyceride')])
ggcorrplot(correlation_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

set.seed(123)  # Set the seed for reproducibility
sample_size <- 10000  # Specify the number of samples you want

# Sample rows from the original data set
sample_data <- dataset[sample(1:nrow(dataset), sample_size), ]

# Split the data into training and test sets
sample_index <- createDataPartition(sample_data$DRK_YN, p = 0.7, list = FALSE)
sample_train_data <- sample_data[sample_index, ]
sample_test_data <- sample_data[-sample_index, ]

# Logistic Regression model with below hyperparameters
logistic_model <- glm(DRK_YN ~ ., data = sample_train_data, family = binomial)
logistic_probabilities <- predict(logistic_model, newdata = sample_test_data, type = "response")

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
logistic_predictions <- ifelse(logistic_probabilities > 0.5, 1, 0)
# Evaluate the accuracy of the tuned logistic regression model
logistic_accuracy <- mean(logistic_predictions == sample_test_data$DRK_YN)
logistic_accuracy

# Create a confusion matrix for further analysis
confusion_matrix <- confusionMatrix(factor(logistic_predictions, levels = c(0, 1)), factor(sample_test_data$DRK_YN, levels = c(0, 1)))

# Print the confusion matrix
print("Confusion Matrix for Logistic Regression:")
print(confusion_matrix)

# K-Nearest Neighbors (KNN) with different values of k
knn1_model <- knn(sample_train_data[, -c(1, 7)], sample_test_data[, -c(1, 7)], sample_train_data$DRK_YN, k = 1)
knn3_model <- knn(sample_train_data[, -c(1, 7)], sample_test_data[, -c(1, 7)], sample_train_data$DRK_YN, k = 3)
knn5_model <- knn(sample_train_data[, -c(1, 7)], sample_test_data[, -c(1, 7)], sample_train_data$DRK_YN, k = 5)
knn20_model <- knn(sample_train_data[, -c(1, 7)], sample_test_data[, -c(1, 7)], sample_train_data$DRK_YN, k = 20)
knn50_model <- knn(sample_train_data[, -c(1, 7)], sample_test_data[, -c(1, 7)], sample_train_data$DRK_YN, k = 50)

# Calculate accuracies for KNN models based on different K values
knn1_accuracy <- mean(knn1_model == sample_test_data$DRK_YN)
knn1_accuracy
knn3_accuracy <- mean(knn3_model == sample_test_data$DRK_YN)
knn3_accuracy
knn5_accuracy <- mean(knn5_model == sample_test_data$DRK_YN)
knn5_accuracy
knn20_accuracy <- mean(knn20_model == sample_test_data$DRK_YN)
knn20_accuracy
knn50_accuracy <- mean(knn50_model == sample_test_data$DRK_YN)
knn50_accuracy

# Confusion matrices for K-Nearest Neighbors (K=50) for further analysis
confusion_matrix_knn50 <- confusionMatrix(factor(knn50_model, levels = c(0, 1)), factor(sample_test_data$DRK_YN, levels = c(0, 1)))
print(confusion_matrix_knn50)

# Random Forest
# Train a Random Forest classifier
rf_model <- randomForest(as.factor(DRK_YN) ~ ., data = sample_train_data)

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = sample_test_data, type = "response")

# Calculate the accuracy of the Random Forest model
rf_accuracy <- mean(rf_predictions == sample_test_data$DRK_YN)
rf_accuracy
# Convert Random Forest predictions to a binary factor
rf_predictions_factor <- factor((rf_predictions), levels = c(0, 1))

# Create a confusion matrix for Random Forest
confusion_matrix_rf <- confusionMatrix(rf_predictions_factor, factor(sample_test_data$DRK_YN, levels = c(0, 1)))
print(confusion_matrix_rf)   

# Naive Bayes
naive_bayes_model <- naiveBayes(DRK_YN ~ ., data = sample_train_data)
naive_bayes_predictions <- predict(naive_bayes_model, newdata = sample_test_data)

# Calculate the accuracy of the Naive Bayes model
naive_bayes_accuracy <- mean(naive_bayes_predictions == sample_test_data$DRK_YN)
naive_bayes_accuracy
# Convert Naive Bayes predictions to a binary factor
naive_bayes_predictions_factor <- factor((naive_bayes_predictions), levels = c(0, 1))

# Create a confusion matrix for Naive Bayes
confusion_matrix_nb <- confusionMatrix(naive_bayes_predictions_factor, factor(sample_test_data$DRK_YN, levels = c(0, 1)))
print(confusion_matrix_nb)   

# Perform hyper parameter tuning with cross-validation
set.seed(123)
param_grid <- expand.grid(
 alpha = 0:1,
 lambda = seq(0.001, 1, length = 10)
)

#training the tuned logistic model
logistic_model_tuned <- train(
 DRK_YN ~ .,
 data = sample_train_data,
 method = "glmnet",
 trControl = trainControl(method = "cv", number = 10),
 tuneGrid = param_grid
)

# Get the best hyper parameters
best_alpha <- logistic_model_tuned$bestTune$alpha
best_lambda <- logistic_model_tuned$bestTune$lambda

# Train the final logistic regression model with the best hyper parameters
final_logistic_model <- glmnet(
 x = model.matrix(DRK_YN ~ ., data = sample_train_data),
 y = sample_train_data$DRK_YN,
 alpha = best_alpha,
 lambda = best_lambda
)

# Make predictions on the test set with the tuned model
logistic_hpt_predictions <- ifelse(predict(final_logistic_model, s = best_lambda, newx = model.matrix(DRK_YN ~ ., data = sample_test_data)) > 0.5, 1, 0)

# Evaluate the accuracy of the tuned logistic regression model
logistic_hpt_accuracy <- mean(logistic_hpt_predictions == sample_test_data$DRK_YN)
logistic_hpt_accuracy

# Create a confusion matrix with tuned logistic model predictions
confusion_matrix <- confusionMatrix(factor(logistic_hpt_predictions, levels = c(0, 1)), factor(sample_test_data$DRK_YN, levels = c(0, 1)))

# Print the confusion matrix
print("Confusion Matrix for tuned Logistic Regression:")
print(confusion_matrix)


# Create a comparison table to compare accuracies of different models
comparison_table <- data.frame(
 Model = c("Logistic Regression", "K-Nearest Neighbors K=1","K-Nearest Neighbors K=3", "K-Nearest Neighbors K=5","K-Nearest Neighbors K=20","K-Nearest Neighbors K=50","Random Boost", "Naive Bayes","Logistic Regression Accuracy (after hyperparameter tuning)"),
 Accuracy = c(logistic_accuracy, knn1_accuracy, knn3_accuracy, knn5_accuracy, knn20_accuracy, knn50_accuracy, rf_accuracy, naive_bayes_accuracy, logistic_hpt_accuracy)
)

# Print the comparison table
print(comparison_table)

# KNN and Naive Bayes Model Prediction Project
Prediction of Alcohol behavior using Logistic, Random boost, KNN, Naive Bayes models and comparing the accuracy metrics

dataset = https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

## Problem Statement

The aim of this project is to predict whether individuals have a drinking habit (`DRK_YN`) using various health metrics like **age**, **height**, **weight**, **cholesterol**, and **blood pressure**. This binary classification problem is tackled using several machine learning algorithms.

## Approach

1. **Data Loading**: 
   - The dataset is loaded from a CSV file, and then split into training and testing sets using the `train_test_split` function.
   
2. **Data Transformation**:
   - We preprocess the data by:
     - **Standard Scaling** to normalize numerical features.
     - **Label Encoding** to transform categorical features like **gender** and the target variable `DRK_YN`.
   
3. **Model Training**:
   - Models trained include:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest**
     - **Naive Bayes**
   - Hyperparameter tuning is performed for the KNN and Random Forest models to optimize performance.

4. **Model Evaluation**:
   - The models are evaluated based on accuracy. The best-performing model is selected based on the evaluation metrics.

## Models Used

- **Logistic Regression**: Chosen for its simplicity and interpretability in binary classification.
- **K-Nearest Neighbors (KNN)**: Chosen for its instance-based learning approach. It is a non-parametric algorithm that makes no assumptions about the distribution of the data.
- **Random Forest**: A robust ensemble method that combines the output of multiple decision trees to make a decision.
- **Naive Bayes**: Chosen for its fast computation and simplicity, assuming feature independence.

### Why these models?

- **Logistic Regression**: Efficient for binary classification tasks.
- **KNN**: Effective for non-linear decision boundaries and easy to implement.
- **Random Forest**: Performs well on both small and large datasets, handling both categorical and continuous variables.
- **Naive Bayes**: Fast to train and works well with high-dimensional datasets.

## Hyperparameter Tuning

- **K-Nearest Neighbors (KNN)**:
  - The value of `K` (number of neighbors) is tuned to find the optimal number of neighbors.
  
- **Random Forest**:
  - The number of trees in the forest (`n_estimators`) is tuned to improve model accuracy.

**Advantages**:
- Hyperparameter tuning improves the model's performance by optimizing key parameters that affect how the model learns from data.

## Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/KNN-NaiveBayes-Prediction.git
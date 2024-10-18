from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_logistic(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, k=50):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train):
    #grid_search = GridSearchCV(model, param_grid, cv=5)

    param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2']
    }

    logistic_cv_model = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=500), param_grid, cv=10)
    logistic_cv_model.fit(X_train, y_train)

    # Best hyperparameters
    best_C = logistic_cv_model.best_params_['C']
    best_penalty = logistic_cv_model.best_params_['penalty']

    # Final logistic model with best parameters
    final_logistic_model = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear', max_iter=1000)
    final_logistic_model.fit(X_train, y_train)
    #final_logistic_predictions = final_logistic_model.predict(X_test)

    # Evaluate final logistic model
    #final_logistic_accuracy = accuracy_score(y_test, final_logistic_predictions)
    #print("Tuned Logistic Regression Accuracy:", final_logistic_accuracy)
    #grid_search.fit(X_train, y_train)
    return final_logistic_model
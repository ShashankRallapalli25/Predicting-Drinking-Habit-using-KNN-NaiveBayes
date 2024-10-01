from sklearn.metrics import accuracy_score
from .data_transformation import get_preprocessed_data
from .Model_training import train_logistic, train_random_forest, train_knn, train_naive_bayes

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def main():
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    
    # Train models
    logistic_model = train_logistic(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Evaluate models
    logistic_acc = evaluate_model(logistic_model, X_test, y_test)
    rf_acc = evaluate_model(rf_model, X_test, y_test)
    knn_acc = evaluate_model(knn_model, X_test, y_test)
    nb_acc = evaluate_model(nb_model, X_test, y_test)
    
    # Compare accuracies
    accuracies = {
        "Logistic Regression": logistic_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Naive Bayes": nb_acc
    }
    
    best_model = max(accuracies, key=accuracies.get)
    
    print("Model Accuracies:", accuracies)
    print("Best Model:", best_model)

if __name__ == "__main__":
    main()
from src.components.data_transformation import get_preprocessed_data
from src.components.data_load import save_split_data, load_data, split_data
from src.components.Model_training import train_logistic, train_random_forest, train_knn, train_naive_bayes, hyperparameter_tuning
from src.components.model_evaluation import evaluate_model


def main():
    dataset = load_data("artifacts/smoking_driking_dataset_Ver01.csv")
    X_train, X_test, y_train, y_test = split_data(dataset)
    save_split_data(X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = get_preprocessed_data()
    #Train models
    y_train = y_train.ravel()  #y_train = y_train.values.flatten()
    logistic_model = train_logistic(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    logistic_gridsearch_cv = hyperparameter_tuning(X_train, y_train)
    
    # Evaluate models
    logistic_acc = evaluate_model(logistic_model, X_test, y_test)
    rf_acc = evaluate_model(rf_model, X_test, y_test)
    knn_acc = evaluate_model(knn_model, X_test, y_test)
    nb_acc = evaluate_model(nb_model, X_test, y_test)
    logistic_grid_acc = evaluate_model(logistic_gridsearch_cv, X_test, y_test)

    # Compare accuracies
    accuracies = {
        "Logistic Regression": logistic_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc,
        "Naive Bayes": nb_acc,
        "Logistic_grid": logistic_grid_acc
     }
    
    best_model = max(accuracies, key=accuracies.get)
    
    print("Model Accuracies:", accuracies)
    print(f"Best Model:{best_model} with accuracy : {accuracies[best_model]}")

if __name__ == "__main__": 
    main()



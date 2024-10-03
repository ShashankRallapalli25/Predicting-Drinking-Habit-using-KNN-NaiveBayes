from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def preprocess_data(X_train, X_test):
    """Apply transformations like scaling and encoding."""
    '''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    '''
    # Example using ColumnTransformer to encode categorical variables
    categorical_features = ['sex']  # Add other categorical feature names here
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    # Use a pipeline to include transformations
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', onehot_encoder, categorical_features)
    ],
    remainder='passthrough'  # This keeps other features untouched
    )
    # Create a pipeline for preprocessing and model training
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

    X_train_scaled = preprocessor.fit_transform(X_train)  # Only scale and encode training data
    X_test_scaled = preprocessor.transform(X_test) 
    return X_train_scaled, X_test_scaled

def encode_labels(y_train, y_test):
    """Encode labels if they are categorical."""
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    return y_train_encoded, y_test_encoded

def get_preprocessed_data():
    """Loads, transforms, and returns preprocessed data."""
    X_train = pd.read_csv('artifacts/X_train.csv')
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_train = pd.read_csv('artifacts/y_train.csv')
    y_test = pd.read_csv('artifacts/y_test.csv')
    
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
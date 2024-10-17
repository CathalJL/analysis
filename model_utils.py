import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_model(model_path):
    """
    Load a model using joblib from a given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")

    model = joblib.load(model_path)
    return model

def save_model_with_attributes(model, file_path):
    """
    Save the model to the specified path.
    """
    joblib.dump(model, file_path)

def train_model(train_data, target_column, model_type='RandomForest'):
    """
    Train a new model based on the provided data and return it.
    """
    # Ensure all column names are string type
    train_data.columns = train_data.columns.astype(str)

    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Choose between rf or xg, TODO: Add more and distinguish use cases
    if model_type == 'RandomForest':
        classifier = RandomForestClassifier(n_estimators=100)
    elif model_type == 'XGBoost':
        classifier = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create and fit the pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', classifier)])
    model.fit(X, y)

    return model

def predict(input_data, model):
    """
    Make predictions using the given model and input data.
    """
    predictions = model.predict(input_data)
    return predictions
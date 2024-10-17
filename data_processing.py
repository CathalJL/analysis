import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder





def handle_missing_values(df):
    """
    Handle missing values automatically by filling:
    - Categorical columns with the mode.
    - Numerical columns with the mean or 0.
    """
    df = df.copy() 

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            # Check if it's a numerical column
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().sum() / df.shape[0] > 0.3:
                    df[col] = df[col].fillna(0)  # Fill with 0 if more than 30% are missing
                else:
                    df[col] = df[col].fillna(df[col].mean())  # Fill with mean otherwise
            else:
                df[col] = df[col].fillna(df[col].mode()[0])  # Fill categorical with mode
    return df

def preprocess_data(input_data, target_column=None, encoders=None, scaler=None, model=None):
    """
    Preprocess the input data:
    - Handle missing or NaN values.
    - Encode categorical variables using provided encoders.
    - Scale numerical features using provided scaler.
    """
    input_data = handle_missing_values(input_data)

    # Convert all column names to strings to avoid type mismatch issues
    input_data.columns = input_data.columns.astype(str)

    # Separate features and target to avoid processing the target column
    if target_column:
        features = input_data.drop(columns=[target_column])
        target = input_data[target_column]
    else:
        features = input_data
        target = None

    if encoders:
        for col, encoder in encoders.items():
            if col in features.columns:
                if isinstance(encoder, OrdinalEncoder):
                    # OrdinalEncoder expects 2D input
                    features[[col]] = encoder.transform(features[[col]])
                else:
                    # For LabelEncoder
                    features[col] = features[col].map(lambda s: encoder.classes_.index(s) if s in encoder.classes_ else -1)
            if col == target_column and target is not None:
                if isinstance(encoder, OrdinalEncoder):
                    target = encoder.transform(target.values.reshape(-1, 1))
                else:
                    target = target.map(lambda s: encoder.classes_.index(s) if s in encoder.classes_ else -1)
    else:
        encoders = {}
        for col in features.select_dtypes(include=['object']).columns:
            # Use OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            features[[col]] = encoder.fit_transform(features[[col]])
            encoders[col] = encoder  # Save the encoder for future use

        if target is not None and (target.dtype == 'object' or target.dtype.name == 'category'):
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            target = encoder.fit_transform(target.values.reshape(-1, 1))
            encoders[target_column] = encoder

    # Scale numerical features
    numerical_cols = features.select_dtypes(include=[np.number]).columns
    if scaler:
        features[numerical_cols] = scaler.transform(features[numerical_cols])
    else:
        scaler = StandardScaler()
        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    # Merge the processed features back with the target column, if provided TODO: Add handling for when theres no column
    if target is not None:
        if isinstance(target, np.ndarray):
            target = pd.Series(target.flatten(), name=target_column)
        processed_data = pd.concat([features, target], axis=1)
    else:
        processed_data = features

    return processed_data, encoders, scaler
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


# This file has deprecated functions that need to be removed. Gui.py is becoming main file


def handle_missing_values(df):
    """
    Handle missing values automatically by filling:
    - categorical columns with the mode.
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


# Load pre - trained ml model
def load_model(model_path):
    """
    Load a model using joblib from a given path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    model = joblib.load(model_path)
    return model




# Why do I have 2 of the same functions? TODO: rename it. it has a different purpose, shouldnt have same name though
def preprocess_data(input_data, target_column=None, encoders=None):
    """
    Preprocess the input data:
    - Handle missing or NaN values.
    - Encode categorical variables using provided encoders.
    - Scale numerical features.
    """
    input_data = handle_missing_values(input_data)

    input_data.columns = input_data.columns.astype(str)

    if target_column:
        features = input_data.drop(columns=[target_column])
        target = input_data[target_column]
    else:
        features = input_data
        target = None

    if encoders:
        for col, le in encoders.items():
            if col in features.columns:
                features[col] = le.transform(features[col])
            if col == target_column and target is not None:
                target = le.transform(target)

    numerical_cols = features.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    if target is not None:
        processed_data = pd.concat([features, target], axis=1)
    else:
        processed_data = features

    return processed_data

def predict(input_data, model):
    """
    Make predictions using the given model and input data.
    """
    predictions = model.predict(input_data)
    return predictions



# Model training
import pickle  # To save and load encoders


def train_model(train_data, target_column, model_type='RandomForest'):
    """
    Train a new model based on the provided data and save it.
    """
    # Ensure all column names are string type
    train_data.columns = train_data.columns.astype(str)
    
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]

    encoders = {}  # Dictionary to store encoders for categorical features

    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le  # Save the encoder for future use

    # Encode the target column if it's categorical
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        encoders[target_column] = target_encoder  # Save the target encoder

    # Choose between random forest or XGBoost TODO: Add more
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'XGBoost':
        model = XGBClassifier(n_estimators=100)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)

    # Save feature names and encoders as model attributes
    model.feature_names_in_ = X.columns.astype(str)
    model.encoders = encoders

    return model

# Main programme block
class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Prediction App")
        self.root.geometry("400x200")


        self.csv_path = ""
        self.model_path = ""
        self.selected_target_column = tk.StringVar()

        tk.Button(root, text="Upload CSV", command=self.upload_csv).pack(pady=10)
        tk.Button(root, text="Select Predefined Model", command=self.select_predefined_model).pack(pady=10)
        tk.Button(root, text="Upload Model File", command=self.upload_model).pack(pady=10)
        tk.Button(root, text="Train New Model", command=self.train_new_model).pack(pady=10)
        tk.Button(root, text="Run Prediction", command=self.run_prediction).pack(pady=10)

    def upload_csv(self):
        self.csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.csv_path:
            messagebox.showinfo("File Selected", f"CSV file loaded: {self.csv_path}")

    def upload_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.joblib *.pkl")])
        if self.model_path:
            messagebox.showinfo("File Selected", f"Model file loaded: {self.model_path}")

    def select_predefined_model(self):
        predefined_models = {
            "Random Forest": "path_to_random_forest_model.joblib",
            "XGBoost": "path_to_xgboost_model.joblib"
        }
        
        model_window = tk.Toplevel(self.root)
        model_window.title("Select a Model")
        model_window.geometry("300x100")
        
        model_var = tk.StringVar(model_window)
        model_var.set("Random Forest")
        
        tk.Label(model_window, text="Choose a predefined model:").pack(pady=10)
        model_dropdown = tk.OptionMenu(model_window, model_var, *predefined_models.keys())
        model_dropdown.pack(pady=10)

        def select_model():
            selected_model = model_var.get()
            self.model_path = predefined_models[selected_model]
            messagebox.showinfo("Model Selected", f"Predefined model loaded: {self.model_path}")
            model_window.destroy()

        tk.Button(model_window, text="Select", command=select_model).pack()

    def train_new_model(self):
        if not self.csv_path:
            messagebox.showwarning("Input Error", "Please upload a CSV file for training")
            return
        
        try:
            data = pd.read_csv(self.csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return
        
        target_column_window = tk.Toplevel(self.root)
        target_column_window.title("Select Target Column")
        target_column_window.geometry("400x200")

        column_names = list(data.columns)

        self.selected_target_column.set(column_names[0])

        tk.Label(target_column_window, text="Select the target column, this is the one you want to make predictions for:").pack(pady=10)
        target_column_dropdown = tk.OptionMenu(target_column_window, self.selected_target_column, *column_names)
        target_column_dropdown.pack(pady=10)

        model_type_var = tk.StringVar(target_column_window)
        model_type_var.set("RandomForest")  # Default to RandomForest

        tk.Label(target_column_window, text="Choose a model type to train:").pack(pady=10)
        model_type_dropdown = tk.OptionMenu(target_column_window, model_type_var, "RandomForest", "XGBoost")
        model_type_dropdown.pack(pady=10)

        def save_model_with_features(model, feature_names, file_path):
            """
            Save the model along with feature names used during training.
            """
            # Adding feature names as an attribute to the model
            model.feature_names_in_ = feature_names
            joblib.dump(model, file_path)

        def confirm_training():
            target_column = self.selected_target_column.get()
            model_type = model_type_var.get()

            try:
                data_processed = preprocess_data(data, target_column)

                if target_column not in data_processed.columns:
                    messagebox.showwarning("Input Error", f"Target column '{target_column}' not found.")
                    return

                trained_model = train_model(data_processed, target_column, model_type)

                save_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])
                if not save_path:
                    messagebox.showwarning("Save Error", "You must specify a file to save the trained model.")
                    return

                joblib.dump(trained_model, save_path)
                messagebox.showinfo("Success", f"Model trained and saved to {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during training: {e}")

            target_column_window.destroy()

        tk.Button(target_column_window, text="Train Model", command=confirm_training).pack(pady=10)

    def run_prediction(self):
        if not self.csv_path or not self.model_path:
            messagebox.showwarning("Input Error", "Please upload both CSV and model files.")
            return

        try:
            data = pd.read_csv(self.csv_path)

            target_column_window = tk.Toplevel(self.root)
            target_column_window.title("Select Target Column")
            target_column_window.geometry("400x200")

            column_names = list(data.columns)

            selected_target_column = tk.StringVar(target_column_window)
            selected_target_column.set(column_names[0])

            tk.Label(target_column_window, text="Select the target column used during training:").pack(pady=10)
            target_column_dropdown = tk.OptionMenu(target_column_window, selected_target_column, *column_names)
            target_column_dropdown.pack(pady=10)

            def confirm_prediction():
                target_column = selected_target_column.get()

                data.columns = data.columns.astype(str)

                model = load_model(self.model_path)
                encoders = getattr(model, 'encoders', None)  # Retrieve saved encoders, if any

                data_processed = preprocess_data(data, target_column, encoders)

                if target_column in data_processed.columns:
                    data_processed_no_target = data_processed.drop(columns=[target_column])
                else:
                    data_processed_no_target = data_processed

                # Debugging only: Print the columns used for prediction
                print("Columns used for prediction:")
                print(data_processed_no_target.columns)

                if hasattr(model, 'feature_names_in_'):
                    expected_features = set(str(f) for f in model.feature_names_in_)
                    prediction_features = set(str(f) for f in data_processed_no_target.columns)

                    if expected_features != prediction_features:
                        missing_in_data = expected_features - prediction_features
                        extra_in_data = prediction_features - expected_features
                        raise ValueError(f"Feature name mismatch:\nMissing in input data: {missing_in_data}\nExtra in input data: {extra_in_data}")

                predictions = predict(data_processed_no_target, model)

                output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if not output_file:
                    messagebox.showwarning("Save Error", "You must specify a file to save the predictions.")
                    return

                predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
                predictions_df.to_csv(output_file, index=False)

                messagebox.showinfo("Success", f"Predictions saved to {output_file}")

                target_column_window.destroy()

            tk.Button(target_column_window, text="Run Prediction", command=confirm_prediction).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPredictionApp(root)
    root.mainloop()

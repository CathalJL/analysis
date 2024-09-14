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



# Error handling for missing values
def handle_missing_values(df):
    """
    Handle missing values automatically by filling:
    - categorical columns with the mode.
    - Numerical columns with the mean or 0.
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isnull().sum() / df.shape[0] > 0.3:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
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


# Preprocess the input dataset
def preprocess_data(input_data):
    """
    Preprocess the input data:
    - Handle missing or Nan values
    """
    # Handle missing or NaN values
    input_data = handle_missing_values(input_data)

    # TODO: Add addtional preprocessing steps like scaling or encoding
    return input_data


# Function to predict using the model
def predict(input_data, model):
    """
    Make predictions using the given model and input data.
    """
    predictions = model.predict(input_data)
    return predictions



# Model training
def train_model(train_data, target_column, model_type='RandomForest'):
    """
    Train a new model based on the provided data and save it
    """
    X = train_data.drop(columns=[target_column])
    y = train_data[target_column]

    # Choose between random forest or XGBoost
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'XGBoost':
        model = XGBClassifier(n_estimators=100)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X, y)

    return model

# Main programme block
class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Prediction App")
        self.root.geometry("400x200")

        # Variables to hold paths
        self.csv_path = ""
        self.model_path = ""

        # Add buttons for file selection
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
        # You can define paths to predefined models here
        predefined_models = {
            "Random Forest": "path_to_random_forest_model.joblib",
            "XGBoost": "path_to_xgboost_model.joblib"
        }
        
        # Create a simple dropdown to choose a model
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
        
        # Load the CSV to get column names
        try:
            data = pd.read_csv(self.csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")
            return
        
        target_column_window = tk.Toplevel(self.root)
        target_column_window.title("Select Target Column")
        target_column_window.geometry("400x200")

        # Get the column names from the dataset
        column_names = list(data.columns)

        selected_target_column = tk.StringVar(target_column_window)
        selected_target_column.set(column_names[0])


        tk.Label(target_column_window, text="Select the target column, this is the one you want to make predictions for:").pack(pady=10)
        target_column_dropdown = tk.OptionMenu(target_column_window, selected_target_column, *column_names)
        target_column_dropdown.pack(pady=10)


        # Create model type selection window
        model_type_var = tk.StringVar(target_column_window)
        model_type_var.set("RandomForest")  # Default to RandomForest

        tk.Label(target_column_window, text="Choose a model type to train:").pack(pady=10)
        model_type_dropdown = tk.OptionMenu(target_column_window, model_type_var, "RandomForest", "XGBoost")
        model_type_dropdown.pack(pady=10)


        def confirm_training():
            target_column = selected_target_column.get()
            model_type = model_type_var.get()

            try:
                data_processed = preprocess_data(data)

                if target_column not in data_processed.columns:
                    messagebox.showwarning("Input Error", f"Target column '{target_column}' not found.")
                    return

                # Train the model
                trained_model = train_model(data_processed, target_column, model_type)

                # Save the trained model
                save_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])
                if not save_path:
                    messagebox.showwarning("Save Error", "You must specify a file to save the trained model.")
                    return

                joblib.dump(trained_model, save_path)
                messagebox.showinfo("Success", f"Model trained and saved to {save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during training: {e}")

            # Close the window after training
            target_column_window.destroy()

        tk.Button(target_column_window, text="Train Model", command=confirm_training).pack(pady=10)


    def run_prediction(self):
        if not self.csv_path or not self.model_path:
            messagebox.showwarning("Input Error", "Please upload both CSV and model files.")
            return

        try:
            # Load the CSV file
            data = pd.read_csv(self.csv_path)
            data_processed = preprocess_data(data)

            # Create a new window to select the target column
            target_column_window = tk.Toplevel(self.root)
            target_column_window.title("Select Target Column")
            target_column_window.geometry("400x200")

            # Get the column names from the dataset
            column_names = list(data.columns)

            # Tkinter StringVar to hold the selected column
            selected_target_column = tk.StringVar(target_column_window)
            selected_target_column.set(column_names[0])  # Default to the first column

            # Create a label and dropdown (OptionMenu) for selecting the target column
            tk.Label(target_column_window, text="Select the target column used during training:").pack(pady=10)
            target_column_dropdown = tk.OptionMenu(target_column_window, selected_target_column, *column_names)
            target_column_dropdown.pack(pady=10)

            # Define function to confirm the prediction
            def confirm_prediction():
                target_column = selected_target_column.get()

                # Preprocess data and ensure the target column is removed during prediction
                if target_column in data_processed.columns:
                    data_processed_no_target = data_processed.drop(columns=[target_column])

                try:
                    # Load the model
                    model = load_model(self.model_path)

                    # Make predictions
                    predictions = predict(data_processed_no_target, model)

                    # Save the output to a CSV file
                    output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                    if not output_file:
                        messagebox.showwarning("Save Error", "You must specify a file to save the predictions.")
                        return

                    predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
                    predictions_df.to_csv(output_file, index=False)

                    messagebox.showinfo("Success", f"Predictions saved to {output_file}")

                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred during prediction: {e}")

                # Close the window after prediction
                target_column_window.destroy()

            # Button to confirm the selected target column and run the prediction
            tk.Button(target_column_window, text="Run Prediction", command=confirm_prediction).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


# Running the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPredictionApp(root)
    root.mainloop()

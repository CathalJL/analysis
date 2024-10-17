import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from model_utils import load_model, train_model, predict, save_model_with_attributes


"""
Need to make it less choppy and not require dragging
"""

class ModelPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Prediction App")
        self.root.geometry("600x400")  # Increased window size - make this dynamic for screen size

        # Variables to hold paths and selected target column
        self.csv_path = ""
        self.model_path = ""
        self.target_column = None  # Initialize target column variable

        tk.Button(root, text="Upload CSV", command=self.upload_csv, width=20).pack(pady=5)
        tk.Button(root, text="Upload Model File", command=self.upload_model, width=20).pack(pady=5)
        tk.Button(root, text="Train New Model", command=self.train_new_model, width=20).pack(pady=5)
        tk.Button(root, text="Run Prediction", command=self.run_prediction, width=20).pack(pady=5)

        # Frame to display predictions and plots
        self.display_frame = tk.Frame(root)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

    def upload_csv(self):
        self.csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.csv_path:
            messagebox.showinfo("File Selected", f"CSV file loaded: {self.csv_path}")

    def upload_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("Model files", ("*.joblib", "*.pkl"))])
        if self.model_path:
            messagebox.showinfo("File Selected", f"Model file loaded: {self.model_path}")

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

        column_names = list(data.columns)

        # Tkinter StringVar to hold the selected column
        selected_target_column = tk.StringVar(target_column_window)
        selected_target_column.set(column_names[0])  # Default to the first column

        tk.Label(target_column_window, text="Select the target column for training:").pack(pady=10)
        target_column_dropdown = tk.OptionMenu(target_column_window, selected_target_column, *column_names)
        target_column_dropdown.pack(pady=10)

        model_type_var = tk.StringVar(target_column_window)
        model_type_var.set("RandomForest")  # Default to RandomForest

        tk.Label(target_column_window, text="Choose a model type to train:").pack(pady=10)
        model_type_dropdown = tk.OptionMenu(target_column_window, model_type_var, "RandomForest", "XGBoost")
        model_type_dropdown.pack(pady=10)

        def confirm_training():
            target_column = selected_target_column.get()
            model_type = model_type_var.get()

            try:
                if target_column not in data.columns:
                    messagebox.showwarning("Input Error", f"Target column '{target_column}' not found.")
                    return

                # Train the model
                trained_model = train_model(data, target_column, model_type)

                # Save the trained model
                save_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                         filetypes=[("Joblib files", "*.joblib")])
                if not save_path:
                    messagebox.showwarning("Save Error", "You must specify a file to save the trained model.")
                    return

                save_model_with_attributes(trained_model, save_path)
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
            data.columns = data.columns.astype(str)

            model = load_model(self.model_path)

            # Create a new window to select the target column TODO: Change this from
            # creating a new window to doing it inline
            target_column_window = tk.Toplevel(self.root)
            target_column_window.title("Select Target Column")
            target_column_window.geometry("400x200")

            column_names = list(data.columns)

            # Tkinter StringVar to hold the selected column
            selected_target_column = tk.StringVar(target_column_window)
            selected_target_column.set(column_names[0])  # Default to the first column

            tk.Label(target_column_window, text="Select the target column used during training:").pack(pady=10)
            target_column_dropdown = tk.OptionMenu(target_column_window, selected_target_column, *column_names)
            target_column_dropdown.pack(pady=10)

            def confirm_prediction():
                target_column = selected_target_column.get()
                self.target_column = target_column 

                # Ensure the target column is removed during prediction cause it screws it up
                if target_column in data.columns:
                    X = data.drop(columns=[target_column])
                else:
                    X = data

                try:
                    predictions = predict(X, model)

                    # Include the target column in the data if available TODO: Add handling for no tgt in other df
                    if target_column in data.columns:
                        result_df = data.copy()
                    else:
                        result_df = X.copy()

                    result_df = result_df.reset_index(drop=True)

                    # Add predictions to result_df
                    result_df['Prediction'] = predictions

                    self.display_predictions(result_df)

                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred during prediction: {e}")

                # Close the target column selection window after prediction
                target_column_window.destroy()

            tk.Button(target_column_window, text="Run Prediction", command=confirm_prediction).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_predictions(self, result_df):
        # Clear any previous content in the display frame
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        # Display predictions in a Text widget
        text_widget = tk.Text(self.display_frame, wrap='none')
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.display_frame, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.insert(tk.END, result_df.to_string())

        save_button = tk.Button(self.display_frame, text="Save Predictions to CSV",
                                command=lambda: self.save_predictions(result_df))
        save_button.pack(pady=5)

        plot_button = tk.Button(self.display_frame, text="Show Data Plot",
                                command=lambda: self.show_plot(result_df))
        plot_button.pack(pady=5)

        visualize_button = tk.Button(self.display_frame, text="Visualize Feature Effects",
                                     command=lambda: self.visualize_feature_effects(result_df))
        visualize_button.pack(pady=5)

    def save_predictions(self, result_df):
        # Save the output to a CSV file
        output_file = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV files", "*.csv")])
        if not output_file:
            messagebox.showwarning("Save Error", "You must specify a file to save the predictions.")
            return

        result_df.to_csv(output_file, index=False)
        messagebox.showinfo("Success", f"Predictions saved to {output_file}")

    def show_plot(self, result_df):
        # Create a new window for the plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Data Plot")
        plot_window.geometry("600x500")

        fig, ax = plt.subplots(figsize=(6, 5))

        if 'Prediction' in result_df.columns:
            ax.scatter(result_df.index, result_df['Prediction'], c='blue', label='Predictions')
            ax.set_xlabel('Index')
            ax.set_ylabel('Predicted Value')
            ax.set_title('Predictions')
            ax.legend()
        else:
            # If no predictions, plot the data
            ax.plot(result_df)
            ax.set_title('Data Plot')

        # Embedding the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def visualize_feature_effects(self, result_df):
        try:
            # Check if the target variable is in the result dataframe
            if self.target_column not in result_df.columns:
                messagebox.showwarning("Visualization Error", "Target variable not found in the data.")
                return

            feature_window = tk.Toplevel(self.root)
            feature_window.title("Select Features to Plot")
            feature_window.geometry("400x300")

            tk.Label(feature_window, text="Select features to plot against the target variable:").pack(pady=10)
            feature_columns = [col for col in result_df.columns if col != self.target_column and col != 'Prediction']
            feature_listbox = tk.Listbox(feature_window, selectmode=tk.MULTIPLE)
            for col in feature_columns:
                feature_listbox.insert(tk.END, col)
            feature_listbox.pack(pady=10)

            def plot_selected_features():
                selected_indices = feature_listbox.curselection()
                selected_features = [feature_columns[i] for i in selected_indices]
                if not selected_features:
                    messagebox.showwarning("Visualization Error", "No features selected.")
                    return
                self.show_feature_plots(result_df, selected_features)
                feature_window.destroy()

            tk.Button(feature_window, text="Plot Selected Features", command=plot_selected_features).pack(pady=10)
        
        except Exception as e:
            messagebox.showerror("Visualization Error", f"An error occurred: {e}")


    def show_feature_plots(self, result_df, selected_features):
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Feature Effects on Target Variable")
        plot_window.geometry("800x600")

        num_features = len(selected_features)
        fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 5 * num_features))

        if num_features == 1:
            axes = [axes]

        for ax, feature in zip(axes, selected_features):
            try:
                x_values = result_df[feature]
                y_values = result_df[self.target_column]

                x_values = pd.to_numeric(x_values, errors='coerce')
                y_values = pd.to_numeric(y_values, errors='coerce')

                # Remove NaN values
                mask = ~x_values.isna() & ~y_values.isna()
                x_values = x_values[mask]
                y_values = y_values[mask]

                # Check if data is available after cleaning
                if x_values.empty or y_values.empty:
                    ax.text(0.5, 0.5, f"No valid data for {feature}", fontsize=12, ha='center')
                    ax.set_title(f'{feature} vs. {self.target_column}')
                    ax.axis('off')
                    continue

                ax.scatter(x_values, y_values, alpha=0.7)
                ax.set_xlabel(feature)
                ax.set_ylabel(self.target_column)
                ax.set_title(f'{feature} vs. {self.target_column}')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error plotting {feature}:\n{e}", fontsize=12, ha='center')
                ax.set_title(f'{feature} vs. {self.target_column}')
                ax.axis('off')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(plot_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
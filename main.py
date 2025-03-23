from tkinter import filedialog, Text
import tkinter as tk
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Global variables
le = LabelEncoder()
x_train, x_test, y_train, y_test = None, None, None, None
model = None
dataset = None

# Helper function to log messages
def log_message(text_widget, message):
    text_widget.insert(tk.END, message + "\n")
    text_widget.see(tk.END)

# Upload dataset
def upload_dataset(text):
    global dataset
    filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not filename:
        log_message(text, "No file selected. Please upload a valid dataset.")
        return

    try:
        dataset = pd.read_csv(filename)
        if dataset.empty:
            log_message(text, "The uploaded dataset is empty. Please provide a valid file.")
            return
        log_message(text, f"Dataset loaded successfully: {filename}")
        log_message(text, f"Dataset shape: {dataset.shape}")
    except Exception as e:
        log_message(text, f"Error loading dataset: {str(e)}")

# Preprocess dataset
def preprocess_dataset(text):
    global x_train, x_test, y_train, y_test
    if dataset is None:
        log_message(text, "No dataset uploaded. Please upload a dataset first.")
        return

    try:
        x = dataset.drop(columns=['class'])
        y = le.fit_transform(dataset['class'])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        log_message(text, "Dataset preprocessing completed successfully.")
        log_message(text, f"Training data shape: {x_train.shape}")
        log_message(text, f"Test data shape: {x_test.shape}")
    except Exception as e:
        log_message(text, f"Error during preprocessing: {str(e)}")

# Train DNN model
def train_dnn(text):
    global model
    if x_train is None or y_train is None:
        log_message(text, "Data not preprocessed. Please preprocess the dataset first.")
        return

    try:
        model = Sequential([
            Dense(128, input_dim=x_train.shape[1], activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', 
                     optimizer='adam', 
                     metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        log_message(text, "DNN model trained successfully.")
    except Exception as e:
        log_message(text, f"Error during model training: {str(e)}")

# Generate classification report
def classification_report_gui(text):
    if model is None:
        log_message(text, "Model not trained. Please train the model first.")
        return

    try:
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        log_message(text, "Classification Report:\n" + report)
        log_message(text, f"Confusion Matrix:\n{cm}")

        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        log_message(text, f"Precision: {precision:.4f}")
        log_message(text, f"Recall: {recall:.4f}")
    except Exception as e:
        log_message(text, f"Error during classification report generation: {str(e)}")

# Generate ROC graph
def roc_graph(text):
    if model is None:
        log_message(text, "Model not trained. Please train the model first.")
        return

    try:
        y_pred_probs = model.predict(x_test)
        y_test_binarized = np.eye(len(np.unique(y_test)))[y_test]

        for i in range(y_pred_probs.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
        log_message(text, "ROC curve generated successfully.")
    except Exception as e:
        log_message(text, f"Error during ROC graph generation: {str(e)}")

# GUI setup
def main():
    root = tk.Tk()
    root.title("DNN Training Tool")
    root.geometry("1000x700")
    root.configure(bg='#1a1a2e')

    # Header
    header_frame = tk.Frame(root, bg='#16213e', pady=15)
    header_frame.pack(fill=tk.X)

    header_label = tk.Label(
        header_frame,
        text="Deep Neural Network Training Tool",
        font=("Helvetica", 24, "bold"),
        fg='#4ecca3',
        bg='#16213e'
    )
    header_label.pack()

    # Main content
    content_frame = tk.Frame(root, bg='#1a1a2e', pady=30)
    content_frame.pack(fill=tk.BOTH, expand=True)

    # Buttons frame
    button_frame = tk.Frame(content_frame, bg='#16213e', padx=30)
    button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20)

    # Text frame
    text_frame = tk.Frame(content_frame, bg='#1a1a2e', padx=20)
    text_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Text widget
    text = Text(
        text_frame,
        wrap=tk.WORD,
        height=30,
        width=70,
        font=("Consolas", 11),
        bg='#232342',
        fg='#4ecca3',
        padx=15,
        pady=15
    )
    text.pack(fill=tk.BOTH, expand=True)

    # Button style
    button_style = {
        'font': ('Helvetica', 11),
        'width': 25,
        'pady': 12,
        'border': 0
    }

    # Buttons
    buttons = [
        ("Upload Dataset", '#0984e3', lambda: upload_dataset(text)),
        ("Preprocess Dataset", '#00b894', lambda: preprocess_dataset(text)),
        ("Train DNN", '#e17055', lambda: train_dnn(text)),
        ("Generate Classification Report", '#9b59b6', lambda: classification_report_gui(text)),
        ("Generate ROC Graph", '#f39c12', lambda: roc_graph(text))
    ]

    for btn_text, btn_color, btn_command in buttons:
        btn = tk.Button(
            button_frame,
            text=btn_text,
            bg=btn_color,
            fg='white',
            command=btn_command,
            **button_style
        )
        btn.pack(pady=10)

    # Welcome message
    log_message(text, "Welcome to DNN Training Tool!\nPlease start by uploading your dataset.")

    root.mainloop()

if __name__ == "__main__":
    main()

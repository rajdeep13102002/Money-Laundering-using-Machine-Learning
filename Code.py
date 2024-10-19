# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """
    Preprocess the dataset for training.
    
    Parameters:
    - file_path: Path to the dataset CSV file.
    
    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Convert categorical data to numeric (if needed)
    df = pd.get_dummies(df, columns=['transaction_type'], drop_first=True)
    
    # Separate features and target variable
    X = df.drop(columns=['transaction_id', 'flag'])
    y = df['flag']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('../data/transactions.csv')
    print("Data preprocessing completed!")
# src/train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data

def train_model():
    """
    Train a Random Forest model on the preprocessed dataset.
    """
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data('../data/transactions.csv')
    
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the trained model to disk
    joblib.dump(model, '../models/random_forest_model.pkl')
    
    print("Model training completed and saved!")

if __name__ == "__main__":
    train_model()
# src/evaluate.py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_data

def evaluate_model():
    """
    Load the trained model and evaluate it on the test set.
    """
    # Load the preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data('../data/transactions.csv')
    
    # Load the trained model
    model = joblib.load('../models/random_forest_model.pkl')
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print the evaluation results
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == "__main__":
    evaluate_model()
# notebooks/analysis.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../data/transactions.csv')

# Show basic statistics
print(df.describe())

# Visualize the distribution of transaction amounts
plt.figure(figsize=(10, 6))
sns.histplot(df['amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Load Data
def load_data(filename):
    data = pd.read_csv(filename)
    print("Data loaded successfully")
    print("Columns in the DataFrame:", data.columns) 
    return data
data = load_data('C:/Users/Rohit Ashok Yadav/OneDrive/Desktop/Gender_detection/voice.csv')

# Data Preprocessing
def preprocess_data(data):
    # Encode labels
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training complete")
    return model, X_train, X_test, y_train, y_test

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions))
    return predictions

# Plot feature importance (if logistic regression coefficients can be considered as such)
def plot_feature_importance(model, features):
    plt.barh(range(len(model.coef_[0])), model.coef_[0])
    plt.yticks(range(len(model.coef_[0])), features)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.show()

def save_model(model):
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl")

def ensure_features_match(train_features, test_data):
    # Ensure test_data has the same columns as train_features in the same order
    missing_cols = set(train_features) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0  # Add missing columns as zeros or appropriate default value
    test_data = test_data[train_features]  # Reorder columns to match training data
    return test_data

# Main execution function

def main():
    data = load_data('voice.csv')
    X, y = preprocess_data(data)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    X_test = ensure_features_match(X_train.columns, X_test)  # Ensure feature consistency
    save_model(model)

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load the dataset
data = pd.read_csv('data.csv')

# Prepare the data
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the model to a file
with open('new_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as new_model.pkl")

# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and split the features (CGPA, IQ) and target variable (placement); then divide the data into training and testing sets.

2.Standardize the training and testing feature sets using StandardScaler to normalize the data.

3.Define and initialize the logistic regression model parameters (weights and bias), then iteratively update them using gradient descent to minimize the logistic loss (cost function).

4.Evaluate the trained model on the test set by predicting the placements, calculating accuracy, classification report, and plotting the confusion matrix.

5.Use the trained model to assess feature importance through weight values and predict placement probability for new student data after scaling the input features.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('placement.csv')

# Display the first few rows
print("Dataset preview:")
print(data.head())

# Split features and target
X = data[['cgpa', 'iq']].values
y = data['placement'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression using Gradient Descent from scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.costs = []
        
    def sigmoid(self, z):
        # Limit z to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self.sigmoid(z)
        
        # Small value to avoid log(0)
        epsilon = 1e-5
        cost = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost
    
    def fit(self, X, y):
        # Initialize parameters
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)
            
            # Compute gradients
            dw = 1/m * np.dot(X.T, (h - y))
            db = 1/m * np.sum(h - y)
            
            # Update parameters
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
            # Compute and store cost
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                self.costs.append(cost)
                print(f"Iteration {i}: Cost = {cost}")
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Create and train the model
print("\nTraining logistic regression model using gradient descent...")
model = LogisticRegression(learning_rate=0.1, iterations=2000)
model.fit(X_train_scaled, y_train)

# Plot the cost function
plt.figure(figsize=(10, 6))
plt.plot(range(0, len(model.costs) * 100, 100), model.costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function during Training')
plt.grid(True)
plt.show()

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Not Placed', 'Placed'])
plt.yticks([0, 1], ['Not Placed', 'Placed'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add numbers to confusion matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), 
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout()
plt.show()

# Feature importance
print("\nFeature Importance:")
feature_names = ['CGPA', 'IQ']
for i, feature in enumerate(feature_names):
    print(f"{feature}: {model.weights[i]:.4f}")

# Make a prediction for a new student
new_student = np.array([[8.5, 120]])  # Example values for cgpa and iq
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

print("\nPrediction for new student:")
print(f"Placement probability: {probability[0]:.4f}")
print(f"Prediction: {'Placed' if prediction[0] == 1 else 'Not Placed'}")
```

## Output:
![logistic regression using gradient descent](sam.png)
<img width="1277" height="645" alt="Screenshot 2025-10-07 111610" src="https://github.com/user-attachments/assets/3b77c1b2-576a-4d8d-95f1-b8153ac79d80" />
<img width="1371" height="712" alt="Screenshot 2025-10-07 111630" src="https://github.com/user-attachments/assets/982c0a53-ec4a-4c3b-b9e0-ae619156cc3f" />
<img width="769" height="588" alt="Screenshot 2025-10-07 111647" src="https://github.com/user-attachments/assets/9e9652e8-077e-4c15-838f-de69c859aec7" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


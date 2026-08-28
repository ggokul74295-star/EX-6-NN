# EXPT 06: Heart Attack Prediction using Multi-Layer Perceptron 

### Programmed by: Gokulan S
### Registered number: 212225230078
### Date: 28/08/2026

## Aim:
To construct a  Multi-Layer Perceptron to predict heart attack using Python

## Algorithm
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.

Step 2:Load the heart disease dataset from a file using pd.read_csv().

Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).

Step 4:Split the dataset into training and testing sets using train_test_split().

Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.

Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.

Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.

Step 8:Make predictions on the testing set using mlp.predict(X_test).

Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().

Step 10:Print the accuracy of the model.

Step 11:Plot the error convergence during training using plt.plot() and plt.show().

## Program
  ```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(X.columns, model.feature_importances_)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## output
Actual Values:
[0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 2 0 2 1 2 2 1 1 0 2 0]

Predicted Values:
[0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 1 0 2 1 1 2 2 1 0 2 0]

Accuracy:
0.9

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30


Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  2  8]]

 <img width="911" height="562" alt="image" src="https://github.com/user-attachments/assets/e415eb3c-59eb-452c-accf-0cbc76caa039" />
 
<img width="1157" height="606" alt="image" src="https://github.com/user-attachments/assets/218cba0d-4aba-4aa4-a719-aa788c724ce0" />

## Results
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.

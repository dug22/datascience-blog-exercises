from perceptron_model import PerceptronModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_dummy_data(num_samples=100):
  X = [] # Input values
  y = [] # Feature labels

  for i in range(num_samples):
    number = np.random.randint(1, 100)
    x = [number] # x is assigned a number
    y_values = 1 if number % 2 != 0 else 0 # if even assign that number with a 1, otherwise 0 if its oddd.
    X.append(x)
    y.append(y_values)

  return np.array(X), np.array(y)

# Method to train the full model
def train_perceptron():
    X, y = generate_dummy_data(num_samples=100)
    model = PerceptronModel(num_features=X.shape[1])
    model.train(X, y, learning_rate=0.1, num_iterations=100)

    predictions = []
    actual_results = []

    # Printing given predictions and actual results
    print("Predictions")
    print("-----------------")
    for i in range(len(X)):
        prediction = model.predict(X[i])
        print(f"Sample # {i} Prediction: {prediction}, Actual: {y[i]}")
        predictions.append(prediction)
        actual_results.append(y[i])

    accuracy = model.accuracy(X, y)
    print("-----------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    df = pd.DataFrame({
        'Sample': list(range(len(X))),
        'Prediction': predictions,
        'Actual': actual_results
    })

    return df, accuracy

df, accuracy = train_perceptron()
matching_count = sum(df['Prediction'] == df['Actual'])
mismatching_count = sum(df['Prediction'] != df['Actual'])
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [matching_count, mismatching_count]
colors = ['lime', 'red']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title(f'Perceptron Prediction Accuracy (Accuracy: {accuracy * 100:.2f}%)')
plt.axis('equal')
plt.show()

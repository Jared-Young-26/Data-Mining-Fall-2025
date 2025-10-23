import pandas as pd
import numpy as np

learning_rate = 0.1
error_threshold = 0.1
weight_adjustment_threshold = 0.03
bias = -0.2

# features and outputs
features = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 0, 0],
    [0, 1, 0]])

output = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [0, 0, 0],
    [0, 0, 0]])

# Neural Network Weights Initialization
weights_1 = np.array([
    [0, -0.1, 0],
    [0.8, 0.6, -0.2],
    [0, 0, 0.7]])

weights_2 = np.array([
    [0.2, 0.6, 0],
    [0.4, 0.6, 0.6],
    [0, 0.4, 0.8]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for sample, (features, output) in enumerate(zip(features, output), start=1):
    
    # Feed Forward
    z1 = np.dot(features, weights_1) + bias
    a1 = sigmoid(z1)
    z2 = np.dot(a1, weights_2) + bias
    a2 = sigmoid(z2)
    
    # Calculate Error
    error = np.sqrt((output - a2) ** 2)
    print(f"\nSample {sample}")
    print(f"\nInput: {features}")
    print(f"\nExpected Output: {output}")
    print(f"\nActual Output: {np.round(a2,2)}")
    print(f"\nError: {np.round(error,2)}")
    
    # Check if updates are needed
    if np.any(np.abs(error) > error_threshold):
        print("\n\nUpdating Weights...")
        
        # Backpropagation
        b2 = error * a2 * (1 - a2) # Error distributed to output layer
        b1 = np.dot(features, b2) * a1 * (1 - a1) # Error distributed to hidden layer

        # Gradient Descent Matrix Calculation
        grad_2 = np.outer(a1, b2) # Gradients of output layer
        grad_1 = np.outer(features, b1) # Gradients of hidden layer
        
        # Adjust the Gradients by the LR & Update the Weights
        weights_2 += learning_rate * grad_2 # Output layer weight adjustment
        weights_1 += learning_rate * grad_1 # Hidden layer weight adjustment

        print(f" Weights updated successfully")
    else:
        print(f" No update (error below threshold)")
    
    print(f" Update W1:\n{np.round(weights_1,2)}")
    print(f"\nUpdate W2:\n{np.round(weights_2,2)}")

# Final weight matricies
print("\nFinal Weight Matricies:")
print(f"W1:\n{np.round(weights_1,2)}")
print(f"W2:\n{np.round(weights_2,2)}")
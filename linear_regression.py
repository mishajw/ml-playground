#!/usr/bin/env python

import numpy as np
import census_data

# Hyperparameters
LEARNING_RATE = 0.0001
TRAINING_PERC = 0.75
ITERATIONS = 100000

def main():
    # Get training and testing data
    (x, y_real, testing_x, testing_y) = census_data.get_data(TRAINING_PERC)

    # Initialise random weights
    w = np.random.rand(x.shape[1])

    for i in range(ITERATIONS):
        # Perform gradient descent on the weights
        w = gradient_desc(x, y_real, w)

        print("Training set cost: %0.4f, Test set cost: %0.4f" \
                % (cost(x, y_real, w), cost(testing_x, testing_y, w)))

def eval(x, w):
    """Evaluate the prediction given data (x) and weights (w)"""
    return np.dot(x, w)

def cost(x, y_real, w):
    """Evaluate the cost of a set of weights"""
    y_est = eval(x, w)
    m = x.shape[0]

    return (1 / (2 * m)) * np.sum(np.power(y_est - y_real, 2))

def gradient_desc(x, y_real, w):
    """Perform a single step of gradient descent and return the new weights"""
    y_est = eval(x, w)
    m = x.shape[0]
    const = LEARNING_RATE * (1/m)
    inner_loop = x.T * (y_est - y_real)
    change = np.sum(np.multiply(const, inner_loop), 1).T
    
    return np.subtract(w, change)

if __name__ == "__main__":
    main()


#!/usr/bin/env python

import numpy as np

# Data variables
DATA_PATH = 'data/USCensus1990_100.data.txt'
DATA_ROWS = 100
AGE_COLUMN = 1

# Hyperparameters
ALPHA = 0.0001
TRAINING_PERC = 0.75
ITERATIONS = 100000

def main():
    # Get training and testing data
    (x, y_real, testing_x, testing_y) = get_data()

    # Initialise random weights
    w = np.random.rand(x.shape[1])

    for i in range(ITERATIONS):
        # Perform gradient descent on the weights
        w = gradient_desc(x, y_real, w)

        print("Training set cost: %0.4f, Test set cost: %0.4f" \
                % (cost(x, y_real, w), cost(testing_x, testing_y, w)))


def get_data():
    """Get the data from a CSV and format it correctly"""

    all_data = np.genfromtxt(DATA_PATH, delimiter=',')[1:]

    age_column = all_data[:,AGE_COLUMN] # extract the age
    all_data = np.delete(all_data, AGE_COLUMN, 1) # remove the age from parameters
    all_data = np.delete(all_data, 0, 1) # remove the ID
    all_data = np.c_[np.ones(all_data.shape[0]), all_data] # add a column of 1s for linear regression
    
    m = all_data.shape[0]

    training_m = int(m * TRAINING_PERC)

    training_x = all_data[:training_m, :]
    training_y = age_column[:training_m]
    testing_x = all_data[training_m:, :]
    testing_y = age_column[training_m:]

    return (training_x, training_y, testing_x, testing_y)

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
    const = ALPHA * (1/m)
    inner_loop = x.T * (y_est - y_real)
    change = np.sum(np.multiply(const, inner_loop), 1).T
    
    return np.subtract(w, change)

if __name__ == "__main__":
    main()


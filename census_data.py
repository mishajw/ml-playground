#!/usr/bin/env python

import numpy as np

# Data variables
DATA_PATH = 'data/USCensus1990_100.data.txt'
AGE_COLUMN = 1

def get_data(training_perc):
    """Get the data from a CSV and format it correctly"""

    all_data = np.genfromtxt(DATA_PATH, delimiter=',')[1:]

    age_column = all_data[:,AGE_COLUMN] # extract the age
    all_data = np.delete(all_data, AGE_COLUMN, 1) # remove the age from parameters
    all_data = np.delete(all_data, 0, 1) # remove the ID
    all_data = np.c_[np.ones(all_data.shape[0]), all_data] # add a column of 1s for linear regression
    
    m = all_data.shape[0]

    training_m = int(m * training_perc)

    training_x = all_data[:training_m, :]
    training_y = age_column[:training_m]
    testing_x = all_data[training_m:, :]
    testing_y = age_column[training_m:]

    return (training_x, training_y, testing_x, testing_y)


#!/usr/bin/env python

import numpy as np

def main():
    x1 = np.array([1,2,3,4,5,6,7,8,9,0])

    layers = create_nn_layers([10, 5, 3, 1])
    print([l.shape for l in layers])

    results = feed_forward(x1, layers)
    print(results)

def create_nn_layers(layer_sizes):
    layers = []
    
    for i in range(len(layer_sizes) - 1):
        width = layer_sizes[i]
        height = layer_sizes[i+1]
        
        layers.append(np.random.rand(height, width))

    return layers

def feed_forward(x, layers):
    acc = [x[np.newaxis].T]

    for l in layers:
        acc.append(l.dot(acc[-1]))

    return acc

if __name__ == "__main__":
    main()


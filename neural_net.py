#!/usr/bin/env python

import numpy as np

def main():
    x1 = np.array([1,2,3,4,5,6,7,8,9,0])

    layers = create_nn_layers([10, 5, 3, 1])

    results = feed_forward(x1, layers)
    print(results)

class Neuron:
    def forward(self, xs, ws):
        self.xs = xs;
        self.output = xs * ws
        return np.max(np.sum(self.output), 0);

    def backward(self, dl):
        if self.output <= 0:
            return np.zeros(self.xs.shape)
        else:
            return self.xs * dl

class Layer:
    def __init__(self, ws, bs):
        self.ws = ws;
        self.bs = bs;

        assert bs.size == ws.shape[0]

        self.neurons = [Neuron() for i in range(bs.size)]

    @property
    def indices(self):
        return range(len(self.neurons))

    @property
    def rev_indices(self):
        reversed(self.indices)

def create_nn_layers(layer_sizes):
    layers = []
    
    for i in range(len(layer_sizes) - 1):
        width = layer_sizes[i]
        height = layer_sizes[i+1]
        
        layers.append(Layer( \
                np.random.rand(height, width), \
                np.random.rand(height) \
        ))

    return layers

def feed_forward(x, layers):
    acc = x[np.newaxis].T

    for l in layers:

        layer_output = []

        for i in l.indices:
            neuron = l.neurons[i]
            weights = l.ws[i]

            neuron_output = neuron.forward(weights, acc)
            layer_output.append(neuron_output)

        acc = layer_output

    return acc

if __name__ == "__main__":
    main()


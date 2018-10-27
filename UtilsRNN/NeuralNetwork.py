"""
    author : VANDAELE Mathias
"""

import numpy as np
import matplotlib.pyplot as plt

#Calculus of the sigmoid
def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

#Calculus of the sigmoid derivation
def sigmoid_derivative(y):
    return y * (1.0 - y)

#Initialisation of the class (input, output, targets, weights, biais)
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.y          = y
        self.inputdim   = self.input.shape[1]
        self.outputdim   = self.y.shape[1]
        self.recurrentdim = 20
        self.inputWeight = np.random.rand(self.inputdim,self.recurrentdim)
        self.outputWeight   = np.random.rand(self.recurrentdim,self.outputdim)
        self.recurrentWeight   = np.random.rand(self.recurrentdim,self.recurrentdim)

        self.inputWeightUpdate =  np.zeros_like(self.inputWeight)
        self.outputWeightUpdate   =  np.zeros_like(self.outputWeight)
        self.recurrentWeightUpdate   =  np.zeros_like(self.recurrentWeight)

    def feedForward(self):
        output = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.recurrentdim))
        for indexLetter in range(len(self.input)):
            x = self.input[indexLetter]
            layer_1 = sigmoid(np.dot(x,self.inputWeight) + np.dot(layer_1_values[-1],self.recurrentWeight))
            output_layer = sigmoid(np.dot(layer_1,self.outputWeight))
            output.append(output_layer)
        print (output)

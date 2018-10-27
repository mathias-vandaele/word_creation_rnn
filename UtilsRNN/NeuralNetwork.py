"""
    author : VANDAELE Mathias
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def softmax(matrix):
    exp_matrix = np.exp(matrix)
    return (exp_matrix / np.sum(exp_matrix))

def tanh(z) :
    return np.tanh(z)

def tanh_derivative(y):
    return 1 - np.square(y)

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
        self.nbASCII = 255
        self.inputWeight = np.random.rand(self.recurrentdim, self.nbASCII)
        self.outputWeight   = np.random.rand(self.nbASCII, self.recurrentdim)
        self.recurrentWeight   = np.random.rand(self.recurrentdim,self.recurrentdim)



    def feedForward(self):
        self.layer_1_values = list()
        self.output_values = list()
        self.layer_1_values.append(np.zeros(self.recurrentdim))
        for indexLetter in range(len(self.input)):
            inputNormalise = np.zeros(self.nbASCII)
            inputNormalise[self.input[indexLetter]] = 1
            layer_1 = np.dot(self.inputWeight, inputNormalise) + np.dot(self.recurrentWeight, self.layer_1_values[-1])
            layer_1_actif = tanh(layer_1)
            layer_output = np.dot(self.outputWeight, layer_1)
            self.layer_1_values.append(layer_1_actif)
            self.output_values.append(layer_output)

    def prediction(self):
        self.feedForward()
        results =  [np.argmax(softmax(layer)) for layer in self.output_values]
        return results

    def loss_calculation(self):
        assert len(self.input) == len(self.y)
        self.feedForward()
        loss = 0.0
        for i, layer_output in enumerate(self.output_values):
            probs = softmax(layer_output)
            loss += -np.log(probs[int(self.y[i])])
        return loss/len(self.y)

    def backpropagation_throught_time(self):
        T = len(self.y)
        inputWeightUpdate =  np.zeros_like(self.inputWeight)
        outputWeightUpdate   =  np.zeros_like(self.outputWeight)
        recurrentWeightUpdate   =  np.zeros_like(self.recurrentWeight)
        self.feedForward()
        delta_o = [softmax(layer) for layer in self.output_values]

        for t in range(T):
            delta_o[t][int(self.y[t])] -= 1.

        for t in np.arange(T)[::-1]:
            print (np.asarray(delta_o[t]))
            recurrentWeightUpdate += np.outer(np.asarray(delta_o[t]), np.asarray(self.layer_1_values[t].T))

            #delta_t = self.V.T.dot(delta_o[t]) * (1 - (self.layer_1_values[t] ** 2))

"""
    author : VANDAELE Mathias
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def softmax(matrix):
    exp_matrix = np.exp(matrix)
    return (exp_matrix / np.sum(exp_matrix))

#Initialisation of the class (input, output, targets, weights, biais)
class NeuralNetwork:
    def __init__(self, x, y):
        np.random.seed(10)
        self.input      = x
        self.y          = y
        self.recurrentdim = 20
        self.nbASCII = 255
        self.inputWeight = np.random.rand(self.recurrentdim, self.nbASCII)
        self.outputWeight   = np.random.rand(self.nbASCII, self.recurrentdim)
        self.recurrentWeight   = np.random.rand(self.recurrentdim,self.recurrentdim)
        self.bptt_truncate = 100
        self.learning_rate = 0.01

    def feedForward(self, x):
        iterations = len(x)
        layer_1_activated = np.zeros((iterations + 1, self.recurrentdim))
        layer_1_activated[-1] = np.zeros(self.recurrentdim)
        output = np.zeros((iterations, self.nbASCII))

        for iteration in np.arange(iterations):
            layer_1_activated[iteration] = np.tanh(self.inputWeight[:,int(x[iteration])] + self.recurrentWeight.dot(layer_1_activated[iteration-1]))
            output[iteration] = softmax(self.outputWeight.dot(layer_1_activated[iteration]))
        return output, layer_1_activated

    def prediction(self, x):
        output, layer_1_activated = self.feedForward(x)
        results = np.argmax(output, axis=1)
        return results

    def loss_calculation(self):
        nb_entry = np.sum((len(y_i) for y_i in self.y))
        loss = 0
        for i in np.arange(len(self.y)):
            output, layer_1_activated = self.feedForward(self.input[i])
            correct_word_predictions = output[np.arange(len(self.y[i])), self.y[i]]
            loss += -1 * np.sum(np.log(correct_word_predictions))
        return loss/nb_entry


    def backpropagation_throught_time(self, x, y):
        output, layer_1_activated = self.feedForward(x)
        iterations = len(y)

        inputWeight_update = np.zeros(self.inputWeight.shape)
        outputWeight_update = np.zeros(self.outputWeight.shape)
        recurrentWeight_update = np.zeros(self.recurrentWeight.shape)

        delta_o = output
        delta_o[np.arange(len(y)), y] -= 1.
        for iteration in np.arange(iterations)[::-1]:
            outputWeight_update += np.outer(delta_o[iteration], layer_1_activated[iteration].T)
            delta_t = self.outputWeight.T.dot(delta_o[iteration]) * (1 - (layer_1_activated[iteration] ** 2))

            for bptt_step in np.arange(max(0, iteration-self.bptt_truncate), iteration+1)[::-1]:
                recurrentWeight_update += np.outer(delta_t, layer_1_activated[bptt_step-1])
                inputWeight_update[:,x[bptt_step]] += delta_t

                delta_t = self.recurrentWeight.T.dot(delta_t) * (1 - layer_1_activated[bptt_step-1] ** 2)

        self.inputWeight -= inputWeight_update * self.learning_rate
        self.recurrentWeight -= recurrentWeight_update * self.learning_rate
        self.outputWeight -= outputWeight_update * self.learning_rate

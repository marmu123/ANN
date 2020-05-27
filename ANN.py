import matplotlib.pyplot as plt

from math import exp
from random import random

class NeuralNetwork:
    def __init__(self, noInputs, noHiddenNeurons, noOutputs):
        self.noOutputs=noOutputs
        self.network=[]
        hiddenLayer = [{'weights': [random()/1000 for i in range(noInputs + 1)]} for i in range(noHiddenNeurons)]
        self.network.append(hiddenLayer)
        outputLayer = [{'weights': [random()/1000 for i in range(noHiddenNeurons + 1)]} for i in range(noOutputs)]
        self.network.append(outputLayer)

    def activate(self,weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += (weights[i] * inputs[i])
            # if(activation>5000):
            #     activation=5000
            # elif activation<-5000:
            #     activation=-5000
        return activation

    def sigmoidTransfer(self, activation):
        # if (activation > 700):
        #     return 1
        # elif activation<700:
        #     return 0
        return 1.0 / (1.0 + exp(-activation))

    def forwardPropagate(self, xi):
        inputs = xi
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.sigmoidTransfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def derivative(self, output):
        return output * (1.0 - output)

    def backwardPropagateError(self, expected):
        for i in reversed(range(len(self.network))):
            currentLayer = self.network[i]
            errors = []
            if i != len(self.network) - 1: # if we are not on output layer
                for j in range(len(currentLayer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(currentLayer)):
                    neuron = currentLayer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(currentLayer)):
                neuron = currentLayer[j]
                neuron['delta'] = errors[j] * self.derivative(neuron['output'])

    def updateWeights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def train(self, trainInputs, trainOutputs, learningRate, epochs):
        errors=[]
        for epoch in range(epochs):
            totalError = 0
            for k in range(len(trainInputs)):
                outputs = self.forwardPropagate(trainInputs[k])
                computed = [0 for i in range(self.noOutputs)]
                computed[trainOutputs[k]] = 1
                totalError += sum([(computed[i] - outputs[i]) ** 2 for i in range(len(computed))])
                self.backwardPropagateError(computed)
                self.updateWeights(trainInputs[k], learningRate)
            print('>epoch=%d, error=%.3f' % (epoch, totalError))
            errors.append(totalError)



    def predict_one(self, xi):
        outputs = self.forwardPropagate(xi)
        return outputs.index(max(outputs))

    def predict(self,inputs):
        outputs=[]
        for el in inputs:
            outputs.append(self.predict_one(el))
        return outputs
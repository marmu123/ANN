from dataLoader import loadDigitData,normalisation,splitData
from utils import evalMultiClass
from ANN import NeuralNetwork
import numpy as np

def reshape(data):
    x = []
    for line in data:
        for el in line:
            x.append(el)
    return x


inputs, outputs, outputNames = loadDigitData()
trainIn, trainOut, testIn, testOut = splitData(inputs, outputs)
trainInputs=[reshape(el) for el in trainIn]
testInputs=[reshape(el) for el in testIn]


trainInputsNorm,testInputsNorm=normalisation(trainInputs,testInputs)

nn=NeuralNetwork(noInputs=64,noHiddenNeurons=5,noOutputs=10)
nn.train(trainInputsNorm,trainOut,0.05,100)
predicts=nn.predict(testInputsNorm)
print(predicts)
print(testOut)
accuracy,conf=evalMultiClass(np.array(testOut),predicts)
print("Accuracy:"+ str(accuracy))


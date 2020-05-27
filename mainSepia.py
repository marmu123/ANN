from dataLoader import normalisation,splitData
from utils import evalMultiClass
from ANN import NeuralNetwork
from imgprocessing import getImagesWithAndWithoutSepia
import numpy as np

def reshape(data):
    x = []
    for line in data:
        for el in line:
            x.append(el)
    return x

inputs, outputs, outputNames = getImagesWithAndWithoutSepia()
trainIn, trainOut, testIn, testOut = splitData(inputs, outputs)
trainInputs=[reshape([reshape(e) for e in el]) for el in trainIn]
testInputs=[reshape([reshape(e) for e in el]) for el in testIn]


trainInputsNorm,testInputsNorm=normalisation(trainInputs,testInputs)

nn=NeuralNetwork(noInputs=19800,noHiddenNeurons=5,noOutputs=2)
nn.train(trainInputsNorm,trainOut,0.05,100)
predicts=nn.predict(testInputsNorm)
print(predicts)
print(testOut)
accuracy,conf=evalMultiClass(np.array(testOut),predicts)
print("Accuracy:"+ str(accuracy))


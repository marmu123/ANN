from utils import evalMultiClass
import numpy as np
from imgprocessing import getImagesHappySad
from dataLoader import normalisation,splitData
from sklearn import neural_network


def reshape(data):
    x = []
    for line in data:
        for el in line:
            x.append(el)
    return x


inputs, outputs, outputNames = getImagesHappySad()
trainIn, trainOut, testIn, testOut = splitData(inputs, outputs)
trainInputs=[reshape(el) for el in trainIn]
testInputs=[reshape(el) for el in testIn]


trainInputsNorm,testInputsNorm=normalisation(trainInputs,testInputs)

nn=neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=200, solver='sgd',verbose=10, random_state=1, learning_rate_init=.1)
nn.fit(trainInputsNorm,trainOut)
predicts=nn.predict(testInputsNorm)
print('[',end='')
for el in predicts:
    print(str(el)+', ',end='')
print(']')
print(testOut)
accuracy,conf=evalMultiClass(np.array(testOut),predicts)
print("Accuracy:"+ str(accuracy))
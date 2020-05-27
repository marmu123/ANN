def evalMultiClass(real,computed):
    from sklearn.metrics import confusion_matrix
    conf=confusion_matrix(real,computed)
    accuracy=sum([conf[i][i] for i in range(len(set(real)))])/len(computed)
    return accuracy,conf
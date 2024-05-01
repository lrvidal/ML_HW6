import RiceData as rd
import sklearn.neural_network as nn
from sklearn.metrics import log_loss

riceTrain, riceVal, riceTest = rd.getRiceData()

trainSet = [item[:-1] for item in riceTrain]
trainLabels = [item[-1] for item in riceTrain]

one30Layer = nn.MLPClassifier(hidden_layer_sizes=(30))
two20Layer = nn.MLPClassifier(hidden_layer_sizes=(20, 20))

one30Layer.fit(trainSet, trainLabels)
two20Layer.fit(trainSet, trainLabels)

valSet = [item[:-1] for item in riceVal]
valLabels = [item[-1] for item in riceVal]

one30Layer_pred = one30Layer.predict_proba(valSet)
two20Layer_pred = two20Layer.predict_proba(valSet)

one30LayerEntro = log_loss(valLabels, one30Layer_pred)
two20LayerEntro = log_loss(valLabels, two20Layer_pred)

print("Cross-entropy for 1 layer of 30 units model:", one30LayerEntro)
print("Cross-entropy for 2 layers of 20 units model:", two20LayerEntro)

bestModel = nn.MLPClassifier(hidden_layer_sizes=(30))
bestModel.fit(trainSet + valSet, trainLabels + valLabels)

testSet = [item[:-1] for item in riceTest]
testLabels = [item[-1] for item in riceTest]

ex2_bestModel_Pred = bestModel.predict_proba(testSet)
bestModelEntro = log_loss(testLabels, ex2_bestModel_Pred)
print("Cross-entropy for the best model:", bestModelEntro)
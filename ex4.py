from sklearn.ensemble import AdaBoostClassifier as ada
import RiceData as rd
from sklearn.metrics import log_loss


riceTrain, riceVal, riceTest = rd.getRiceData()

trainSet = [item[:-1] for item in riceTrain]
trainLabels = [item[-1] for item in riceTrain]

validationSet = [item[:-1] for item in riceVal]
validationLabels = [item[-1] for item in riceVal]

testSet = [item[:-1] for item in riceTest]
testLabels = [item[-1] for item in riceTest]

ada20model = ada(n_estimators=20)
ada40model = ada(n_estimators=40)
ada60model = ada(n_estimators=60)

ada20model.fit(trainSet, trainLabels)
ada40model.fit(trainSet, trainLabels)
ada60model.fit(trainSet, trainLabels)

validationAda20Predicts = ada20model.predict_proba(validationSet)
validationAda40Predicts = ada40model.predict_proba(validationSet)
validationAda60Predicts = ada60model.predict_proba(validationSet)

logLossAda20 = log_loss(validationLabels, validationAda20Predicts)
logLossAda40 = log_loss(validationLabels, validationAda40Predicts)
logLossAda60 = log_loss(validationLabels, validationAda60Predicts)

print("Cross Entropy for AdaBoost Model with 20 estimators:", logLossAda20)
print("Cross Entropy for AdaBoost Model with 40 estimators:", logLossAda40)
print("Cross Entropy for AdaBoost Model with 60 estimators:", logLossAda60)

print("Best Model is AdaBoost Model with 20 estimators")

bestModel = ada(n_estimators=20)
bestModel.fit(trainSet + validationSet, trainLabels + validationLabels)

ex4_bestModelPred = bestModel.predict_proba(testSet)
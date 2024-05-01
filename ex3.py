import RiceData as rd
import sklearn.tree as tree
from sklearn.metrics import log_loss


riceTrain, riceVal, riceTest = rd.getRiceData()

trainSet = [item[:-1] for item in riceTrain]
trainLabels = [item[-1] for item in riceTrain]

validationSet = [item[:-1] for item in riceVal]
validationLabels = [item[-1] for item in riceVal]

testSet = [item[:-1] for item in riceTest]
testLabels = [item[-1] for item in riceTest]

giniTreeModel = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
infoGainTreeModel = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

giniTreeModel.fit(trainSet, trainLabels)
infoGainTreeModel.fit(trainSet, trainLabels)

validationGiniPredicts = giniTreeModel.predict_proba(validationSet)
validationInfoGainPredicts = infoGainTreeModel.predict_proba(validationSet)

logLossGiniTree = log_loss(validationLabels, validationGiniPredicts)
logLossInfoGainTree = log_loss(validationLabels, validationInfoGainPredicts)

print("Cross Entropy for Gini Tree Model:", logLossGiniTree)
print("Cross Entropy for Information Gain Tree Model:", logLossInfoGainTree)

print("Best Model is Gini Tree Model")

bestModel = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
bestModel.fit(trainSet + validationSet, trainLabels + validationLabels)

ex3_bestModelPred = bestModel.predict_proba(testSet)




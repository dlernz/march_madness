from sklearn.metrics import confusion_matrix
import pandas as pd 
import numpy as np

class Model:

    def __init__(self):
        self.train = None
        self.test = None
        self.forest = None
        self.trainBaseline = None
        self.testBaseline = None
        self.testNames = None
        self.predictions = None
        self.predictProbs = None
        self.confusionMatrix = None
        self.featureImportances = None
        self.baselineAcc = None
        self.modelAcc = None
        
    def setFeatureImportances(self):
        self.feature_importances = pd.DataFrame(self.forest.feature_importances_,
                                   index = self.train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
    
    def setConfusionMatrix(self):
        self.confusionMatrix = confusion_matrix(self.testBaseline, self.predictions, self.forest.classes_)
    
    def getModelPredictionsAndProbs(self):
        stack = np.column_stack((self.predictions.T, self.testBaseline.T, self.testNames[:,0], self.testNames[:,1], self.testNames[:,2], self.testNames[:,3], self.predictProbs[:,0], self.predictProbs[:,1]))
        return stack[stack[:,0].argsort()]
    
    def setPredictions(self):
        trainLabels = self.trainBaseline
        trainLabels = trainLabels.astype(int)
        testLabels = self.testBaseline
        testLabels = testLabels.astype(int)
        testNames = self.testNames
        trainFeatures = np.array(self.train)
        testFeatures = np.array(self.test)
        self.baselineAcc = 1.0*sum(testLabels) / testLabels.shape[0]

        predictions = self.forest.predict(testFeatures)
        self.predictions = predictions
        self.setConfusionMatrix()
        predictProbs = self.forest.predict_proba(testFeatures)
        self.predictProbs = predictProbs
        self.modelAcc = 1.0*(predictions.shape[0] - sum(predictions ^ testLabels)) / predictions.shape[0]

        
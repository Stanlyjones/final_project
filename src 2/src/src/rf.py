from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import minmax_scale

class MachineLearningAlgo:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=20)
        print("Using RandomForestClassifier Algo") 
        X_train = pd.read_csv('result.csv')
        y_train = X_train["type"]
        del X_train["type"]
        X_train.iloc[:] = minmax_scale(X_train.iloc[:])
        self.clf.fit(X_train, y_train.values.ravel())

    def classify(self, data):
        prediction = self.clf.predict(data)
        #print("Detection Result:", prediction)
        return prediction

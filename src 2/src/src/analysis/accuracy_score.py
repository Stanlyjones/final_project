#Accuracy reference: https://github.com/kshitijved/Support_Vector_Machine

from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import cross_val_score

#step1: Load the data in numpy array
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:6]
y = data[:, 6]

#step2: Split the data to training & test data. Test-size is 0.25(25%) of data
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)


#step3: select the machine learning algorithm

#svm
#clf = svm.SVC()
#clf = svm.SVC(kernel="linear",C=0.025)
#clf = svm.SVC(kernel="linear")
#clf = svm.SVC(gamma=2, C=1)

#Decision Tree
#clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)


#Gaussian Naive Bayes
#clf = GaussianNB()   

#Forests of randomized trees
#clf = RandomForestClassifier(n_estimators=10)

#Extra Tree Classifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)


#neural network classifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

#print(clf)
#step4: Train the ML Algo with training data
clf.fit(x_train, y_train)


#step5: Pass the test data for classify or predict
classifier_predictions = clf.predict(x_test)

#step6. Calculate the accuracy from the the prediction result.
print("Accuracy is ", accuracy_score(y_test, classifier_predictions)*100)


#step7. calculate cross validation score
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("cross-validation score",scores.mean())

'''
References:

https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees
'''
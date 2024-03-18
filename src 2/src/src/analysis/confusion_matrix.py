#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#true negatives is , false negatives is , true positives is and false positives 


from sklearn.metrics import confusion_matrix
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
clf = tree.DecisionTreeClassifier()
#clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)


#Gaussian Naive Bayes
#clf = GaussianNB()   

#Forests of randomized trees
#clf = RandomForestClassifier(n_estimators=10)

#Extra Tree Classifier
#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

#neural network classifier
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)




#step4: Train the ML Algo with training data
clf.fit(x_train, y_train)


#step5: Pass the test data for classify or predict
classifier_predictions = clf.predict(x_test)


#step6. Calculate the confusion matrix

print("Actual ", y_test)
print("predictions ", classifier_predictions)


tn, fp, fn, tp = confusion_matrix(y_test, classifier_predictions).ravel()
print("true negative", tn)
print("false positive", fp)
print("false negative", fn)	
print("true positive",  tp)  



print(confusion_matrix(y_test, classifier_predictions))

'''
#step6. Calculate the accuracy from the the prediction result.
print("Accuracy is ", accuracy_score(y_test, classifier_predictions)*100)


#step7. calculate cross validation score
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("cross-validation score",scores.mean())

'''
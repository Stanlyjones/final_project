from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')
X = data[:, 0:6]
y = data[:, 6]

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

# Select the machine learning algorithm
clf = tree.DecisionTreeClassifier()

# Train the ML Algorithm on the training dataset
clf.fit(X_train, y_train)

# Pass the testing dataset for classification
classifier_predictions = clf.predict(X_test)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, classifier_predictions).ravel()

# Print the confusion matrix components
print("true negative", tn)
print("false positive", fp)
print("false negative", fn)    
print("true positive",  tp)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, classifier_predictions)
print(conf_matrix)

# Calculate and print accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, classifier_predictions)
precision = precision_score(y_test, classifier_predictions)
recall = recall_score(y_test, classifier_predictions)
f1 = f1_score(y_test, classifier_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Display the confusion matrix as a heatmap
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


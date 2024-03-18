#http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
from __future__ import division
import numpy
import os
from sklearn import svm
from collections import deque
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np

data = np.loadtxt(open('result.csv', 'rb'), delimiter=',')

#flow_duration,ip_proto,srcport,dstport,byte_count,packet_count,type
flow_duration = 0
ip_proto = 1
srcport = 2
dstport = 3
byte_count = 4
packet_count = 5



'''
#Graph1 
X = data[:, [byte_count,packet_count]]
y = data[:, 6]
clf = svm.SVC()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('SVM DDoS - Decision Region Boundary', size=16)
plt.xlabel('byte_count')
plt.ylabel('packet_count')
plt.savefig("svm_graph1.png")


#Graph1 
X = data[:, [flow_duration,packet_count]]
y = data[:, 6]
clf = svm.SVC()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('SVM DDoS - Decision Region Boundary', size=16)
plt.xlabel('flow_duration')
plt.ylabel('packet_count')
plt.savefig("svm_graph2.png")

'''


#Graph1 
X = data[:, [byte_count,packet_count]]
y = data[:, 6]
clf = svm.SVC()
clf.fit(X, y)
# Plot Decision Region using mlxtend's awesome plotting function
fig = plt.figure(figsize=(10,8))
fig = plot_decision_regions(X=X, 
                      y=y.astype(int),
                      clf=clf, 
                      legend=2)
plt.title('SVM DDoS - Decision Region Boundary', size=16)
plt.xlabel('byte_count')
plt.ylabel('packet_count')
plt.savefig("svm_graph3.png")
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:12:54 2020

@author: dougl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Read and regrouping data
data = pd.read_csv('fld.txt', header = None)
data = pd.DataFrame.to_numpy(data)

x1 = np.array([])
x2 = np.array([])

n1 = 0
n2 = 0
for i in range(0, len(data)):
    if data[i, 2] == 1:
        d = data[i, (0, 1)]
        x1 = np.append(x1, d)
        n1 = n1 + 1
        x1 = np.reshape(x1, (n1, 2))
        
    else:
        d = data[i, (0, 1)]
        x2 = np.append(x2, d)
        n2 = n2 + 1
        x2 = np.reshape(x2, (n2, 2))

# Calculate the mean for each group
u1 = np.mean(x1, axis = 0)
u2 = np.mean(x2, axis = 0)

# Remove the mean for each group
x1ur = x1 - u1
x2ur = x2 - u2

# Calculate the Cov for each group
cov1 = np.dot(x1ur.T, x1ur)
cov2 = np.dot(x2ur.T, x2ur)

# FLD
sw = cov1 + cov2
w = np.dot(np.linalg.inv(sw), (u1 - u2))

# Plt
plt.figure()
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot([-10000 * w[0], 10000 * w[0]], [-10000 * w[1], 10000 * w[1]], 'g')
ax=plt.gca()
ax.set_xlim(-7.5,12)
ax.set_ylim(-7.5,10)

# Scikitlearn
X = np.concatenate((x1, x2), axis = 0)
y = data[:, 2]
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
p_lda = lda.predict(X)
error_lda = np.sum(abs(p_lda - data[:,2]))
slope_sk = -lda.coef_[0][0]/lda.coef_[0][1]
thresh_sk = -lda.intercept_/lda.coef_[0][1]

x = np.linspace(-4, 4, 50) 
y = thresh_sk + slope_sk * x
plt.plot(x, y, c='b')


# Prediction and Precision
thresh = -0.005
slope = -w[0] / w[1]
y_int = -thresh / w[1]
a=np.sign(np.dot(w, data[:, (0, 1)].T) +thresh)
prediction = (np.sign(np.dot(w, data[:, (0, 1)].T) +thresh) + 1) / 2
error = np.sum(abs(prediction - data[:,2]))
percentage = error / len(data)

#   Plot the FLD line
x = np.linspace(-5,5,100)
y = slope*x + y_int
plt.plot(x, y, '--')
plt.show()
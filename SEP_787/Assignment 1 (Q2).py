# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:23:50 2020

@author: dougl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('spam.xlsx')
data = pd.DataFrame.to_numpy(data)

x1 = np.array([])
x2 = np.array([])

n1 = 0
n2 = 0
for i in range(0, len(data)):
    if data[i, -1] == 1:
        d = data[i, 0:57]
        x1 = np.append(x1, d)
        n1 = n1 + 1
        x1 = np.reshape(x1, (n1, 57))
        
    else:
        d = data[i, 0:57]
        x2 = np.append(x2, d)
        n2 = n2 + 1
        x2 = np.reshape(x2, (n2, 57))

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
ax.set_xlim(-2,4)
ax.set_ylim(-2,6)
plt.show()

# Prediction and Precision
# thresh set to -0.001 for a better percentage in precision(0.101; 0.183 when thresh = 0)
thresh = -0.00247
slope = -w[0] / w[1]
y_int = -thresh / w[1]

prediction = (np.sign(np.dot(w, data[:, 0:57].T) +thresh) + 1) / 2
error = np.sum(abs(data[:,57] - prediction))
percentage = error / len(data)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data[:, 57], prediction)
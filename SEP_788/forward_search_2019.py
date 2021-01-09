import numpy
import math
import random
import matplotlib.pyplot
import sklearn.discriminant_analysis

# create data from a multivariate normal distribution
mean1 = [-3, 0, -1, -2, -4]
mean2 = [3, 1, 0, -2, -4]
cov = [[12, 0, 0, 0, 0],[0, 13, 0, 0, 0],[0, 0, 4, 0, 0],[0, 0, 0, 5, 0],[0, 0, 0, 0, 6]]

x1 = numpy.random.multivariate_normal(mean1,cov, 1000)
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)

X = numpy.concatenate((x1,x2))

Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# training set result without feature selection
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
#training
lda.fit(X,Xc)
#testing
prediction = lda.predict(X)
error = sum(abs(prediction - Xc))
print("total error with all features = ", error)

bestFeature = 100*numpy.ones(3)
for iteration in range(3):
    if iteration == 0:
        Xselection = numpy.zeros((2000,1))
    else:
        Xselection = numpy.concatenate((Xselection, numpy.zeros((2000,1))), axis = 1)

    error  = 10000*numpy.ones(5)
    for feature in range(5):
        # have not used the feature before
        if(not(feature == bestFeature[0] or feature == bestFeature[1] 
           or feature == bestFeature[2])):

            #add a feature to the existing features
            Xselection[:,iteration] = X[:,feature]

            #classify using Xselection
            lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
            lda.fit(Xselection,Xc)
            prediction = lda.predict(Xselection)
            error[feature] = sum(abs(prediction - Xc))

    bestFeature[iteration] = numpy.argmin(error)
    Xselection[:, iteration] = X[:,int(bestFeature[iteration])]

# training set result with feature extraction
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(Xselection,Xc)
prediction = lda.predict(Xselection)
error = sum(abs(prediction - Xc))
print("total error with selected features = ", error)






print("done")
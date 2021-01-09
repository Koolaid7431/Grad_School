import numpy
import math
import random
import matplotlib.pyplot
import sklearn.discriminant_analysis

# create data from a multivariate normal distribution
mean1 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
mean2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cov = [[3, 1, 1, 1, 0, 0, 0, 0, 0,  0],
       [1, 4, 0, 0, 0, 0, 0, 0, 0,  0],
       [1, 0, 4, 0, 0, 0, 0, 0, 0,  0],
       [1, 0, 0, 5, 0, 0, 0, 0, 0,  0],
       [0, 0, 0, 0, 6, 0, 0, 0, 0,  0],
       [0, 0, 0, 0, 0, 10, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 11, 0, 0, 0],
       [0, 0, 4, 0, 0, 0,  0, 3, 0, 0],
       [0, 0, 0, 5, 0, 0,  0, 0, 2, 0],
       [0, 0, 0, 0, 0, 0,  0, 0, 0, 6],]

# make sure the matrix is symmetric positive definite
cov = numpy.dot(cov, numpy.transpose(cov))

x1 = numpy.random.multivariate_normal(mean1,cov, 1000)
x2 = numpy.random.multivariate_normal(mean2,cov, 1000)

X = numpy.concatenate((x1,x2))

Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# PCA
Xmc = X - numpy.mean(X)
D,E = numpy.linalg.eig(numpy.dot(Xmc.T,Xmc))

sortIndex = numpy.argsort(D)

ESorted = numpy.zeros((10,10))
index = 0
for i in range(0,10):
    ESorted[:,index] = E[:,sortIndex[i]]
    index = index + 1

meanSquareError = numpy.zeros(10,)
classificationError = numpy.zeros(10,)
ySorted = numpy.dot(X,ESorted)
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
for numDims in range(5,11): 
    
    # reconstruction
    yReduced = ySorted[:,0:numDims]
    EReduced = ESorted[:,0:numDims]
    XReconstructed = numpy.dot(yReduced, numpy.transpose(EReduced))
    meanSquareError[10 - numDims] = sum(sum((XReconstructed - X)**2))/2000

    # classification
    #training
    lda.fit(yReduced,Xc)
    #testing
    prediction = lda.predict(yReduced)
    classificationError[10 - numDims] = sum(prediction != Xc) # sum(prediction != Xc)
    
n = numpy.linspace(0,5,6)
matplotlib.pyplot.plot(n,meanSquareError[0:6])
matplotlib.pyplot.title("PCA")
matplotlib.pyplot.xlabel("Number of Dimensions Removed")
matplotlib.pyplot.ylabel("Mean Square Error")
matplotlib.pyplot.show()

matplotlib.pyplot.plot(n,classificationError[0:6])
matplotlib.pyplot.title("PCA")
matplotlib.pyplot.xlabel("Number of Dimensions Removed")
matplotlib.pyplot.ylabel("Number of Errors")
matplotlib.pyplot.show()

#use backward search to remove columns 
#so that the remaining provide the least error

minError = 1000000*numpy.ones(6,)
# classification with no columns removed
#training
lda.fit(ySorted,Xc)
#testing
prediction = lda.predict(ySorted)
minError[0] = sum(prediction != Xc)

#find the column to remove that provides the lowest error
minErrorColumn = 0
ySelected = ySorted
numCols = 10
for iteration in range(1,6):
    for column in range(numCols):
        yReduced = numpy.delete(ySelected,column,1)
        # classification
        #training
        lda.fit(yReduced,Xc)
        #testing
        prediction = lda.predict(yReduced)
        classificationError = sum(prediction != Xc)
        if classificationError < minError[iteration]:
            minError[iteration] = classificationError
            minErrorColumn = column

    numCols = numCols - 1
    ySelected = numpy.delete(ySelected,minErrorColumn,1)


matplotlib.pyplot.plot(n,minError[0:6])
matplotlib.pyplot.title("Backward Search")
matplotlib.pyplot.xlabel("Number of Dimensions Removed")
matplotlib.pyplot.ylabel("Number of Errors")
matplotlib.pyplot.show()

print("done")
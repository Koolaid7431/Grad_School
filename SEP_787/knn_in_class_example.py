import numpy

X = numpy.loadtxt("Data_for_UCI_named.csv", delimiter = ',')

Xtrain = X[0:7500,0:13]
Xtest = X[7500:10000,0:13]
Xctrain = X[0:7500,13]
Xctest = X[7500:10000,13]

k = 49
dist = numpy.zeros(2500)
errors = numpy.zeros(2500)
for i in range(2500):
    dist = numpy.sum((Xtrain - Xtest[i,:])**2,axis = 1)**0.5
    sortIndex = numpy.argsort(dist)
    bestLabels = Xctrain[sortIndex[0:k]]
    prediction = (sum(bestLabels) > k/2.0)*1.0
    errors[i] = (Xctest[i] != prediction)*1.0

print("total errors = ", numpy.sum(errors))



print("done")
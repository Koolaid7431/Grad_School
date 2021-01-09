import numpy
import random

b = numpy.array([1, 2, 3, 4, 5, 6])

X = 10*numpy.random.random((500, 5))-5

Z = numpy.concatenate((numpy.ones((500,1)), X),axis = 1)

e = numpy.random.normal(0,9,500)

# model to generate the data
y = numpy.dot(Z,b) + e

I = 0.01 * numpy.identity(6)

#ridge regression
bestimate = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(Z.T,Z) + I),Z.T),y)

bestimate2 = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(Z.T,Z)),Z.T),y)

print("done")



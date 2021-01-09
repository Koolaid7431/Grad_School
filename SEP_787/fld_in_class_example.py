import numpy
import random
import math
import matplotlib.pyplot

#implement fld w = Sw-1*(u1 - u2)

mean1 = [0,3]
mean2 = [4,0]
cov = [[2,0],[0,3]]

#data from class 1
x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)
X = numpy.concatenate((x1,x2))

matplotlib.pyplot.scatter(x1[:,0],x1[:,1], c = 'r', marker = '.')
matplotlib.pyplot.scatter(x2[:,0],x2[:,1], c = 'b', marker = '.')

matplotlib.pyplot.ion()
matplotlib.pyplot.show()

xlabels = numpy.ones(1000)
xlabels = numpy.concatenate((xlabels,numpy.zeros(1000)))

#calculate what we need for fld
#class means

u1 = numpy.mean(x1,0)
u2 = numpy.mean(x2,0)

# remove means from classes
x1mc = x1 - u1
x2mc = x2 - u2

#calculate covariance matrices
S1 = numpy.dot(x1mc.T, x1mc)
S2 = numpy.dot(x2mc.T, x2mc)

Sw = S1 + S2

w = numpy.dot(numpy.linalg.inv(Sw),(u1 - u2))

matplotlib.pyplot.plot([-3000*w[0],3000*w[0]], [-3000*w[1],3000*w[1]], 'g--')


#prediction
threshold = 0.01
predictions = (numpy.sign(numpy.dot(w,X.T) + threshold) + 1)/2

error = sum(predictions != xlabels)

errorIndex = numpy.argwhere(predictions != xlabels)

errorPts = X[errorIndex]

errorPts = numpy.squeeze(errorPts)

matplotlib.pyplot.scatter(errorPts[:,0], errorPts[:,1],c = 'g', marker = 'o')





print("done")


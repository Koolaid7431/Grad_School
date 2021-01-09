import numpy
import matplotlib.pyplot
import os 
import pandas

print('test2')
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# upload data
#data = numpy.loadtxt('dataA.txt')
#print(data)
data= pandas.read_excel("dataA.xlsx", header = None)
datanp = pandas.DataFrame.to_numpy(data)
print (data)

#create data from a multivariate normal distribution
mean1 = [-3, 0]

mean2 = [3, 0]

cov1 = [[3,0],[0,2]]

cov2 = [[2,0], [0,3]]

x1 = numpy.random.multivariate_normal(mean1, cov1, 1000)
x2 = numpy.random.multivariate_normal(mean2, cov2, 1000)

matplotlib.pyplot.scatter(x1[:,0], x1[:,1], c = 'b', marker = '.')
matplotlib.pyplot.scatter(x2[:,0], x2[:,1], c = 'r', marker = '.')

matplotlib.pyplot.ion()
matplotlib.pyplot.show()

print(numpy.mean(x1, axis = 0))
print(numpy.mean(x2, axis = 0))

print(numpy.cov(numpy.transpose(x1)))

x1MeanCentered = x1 - mean1
print(numpy.mean(x1MeanCentered,axis = 0))

print(numpy.dot(numpy.transpose(x1MeanCentered), x1MeanCentered)/999)



print("done")
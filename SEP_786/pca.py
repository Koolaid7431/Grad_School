
import numpy

mean1 = [-3, 0, 3]

cov1 = [[3,1,0],[1,2,0],[0,0,1]]


X = numpy.random.multivariate_normal(mean1, cov1, 1000)
Xc = X - numpy.mean(X)

D,E = numpy.linalg.eig(numpy.dot(Xc.T,Xc))
Dsr = numpy.sqrt(D)

U,S,V = numpy.linalg.svd(Xc)

yeigs = numpy.dot(Xc,E)

ycov1 = numpy.dot(yeigs.T,yeigs)

sortIndex = numpy.argsort(D)

newE = numpy.zeros((3,3))
index = 0
for i in range(0,3):
    newE[:,index] = E[:,sortIndex[i]]
    index = index + 1
    
ynew = numpy.dot(Xc,newE)

print("done")
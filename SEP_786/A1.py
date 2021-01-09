#!/usr/bin/env python
# coding: utf-8

# # Assignment 1
# ### CHEM ENG/ SEP 786
# ### Mohammad Kashif Siddiqui - 0755452
# 

# ### Headers

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas
from IPython.display import Image


# ### Checking paths

# In[2]:


print('test1')
dir_path = os.getcwd()
print(dir_path)


# # Question 1
# #### Using the Excel file dataA.xlsx, which contains a 500x3 data matrix (500 data points with 3 attributes), calculate both the mean and the covariance matrix.
# 

# ### Uploading & Checking Data

# In[3]:


dataA = np.loadtxt('dataA.csv', delimiter=',') 
print("dataA:")
print(dataA)
type(dataA)


# ### Mean Matrix
# 
# ##### Process - Finding the means of the columns (axis = 0)

# In[4]:


mean = np.mean(dataA,axis=0)
print('mean of dataA:')
print(mean)


# ### Covariance Matrix
# 
# ##### Process - As discussed in lecture, applying the covariance forumla: cov = (X^T*X)/n-1, Also try out np.cov(data ^T) to confirm.
# ##### Result - np.cov function has far more significant digits by default, and slightly different results --- find out why?! 
# ##### Explaination - the formula np.dot(np.transpose(dmm),dmm)/len(dmm)-1 was missing brackets around (len(dmm)-1) giving a slight offset

# In[9]:


dmm = dataA- mean #calculating data minus mean
print('dmm:')
print(dmm)

cov = np.dot(np.transpose(dmm),dmm)/(len(dmm)-1)
print('cov matrix:')
print(cov)

cov2 = np.cov(dmm.T)
print('cov2 matrix:')
print(cov2)
print('cov2 shape:', cov2.shape)


# In[ ]:





# # Question 2
# #### Using the Excel file dataB.xlsx, which contains a 500x10 data matrix (500 data points with 10 attributes), calculate both the mean and the covariance matrix.

# ### Uploading & Checking Data

# In[10]:


dataB = np.loadtxt('dataB.csv', delimiter=',') 
print("dataB:")
print(dataB)
print(dataB.shape)
type(dataB)


# ### Mean Matrix
# 
# ##### Process - Finding the means of the columns (axis = 0)

# In[11]:


meanB = np.mean(dataB,axis=0)
print('mean of dataB:')
print(meanB)
meanB.shape


# ### Covariance Matrix
# 
# ##### Process - As discussed in lecture, applying the covariance forumla: cov = (X^T*X)/n-1 , Also try out np.cov(data ^T) to confirm
# ##### Result - np.cov function has far more significant digits by default, and slightly different results --- find out why?!
# ##### Explanation - The numbers look different and according to the np.array_equal but according to np.allclose(), the two matrices are similar enough. Same error as before, missing brackets. - Issue considered resolved.

# In[12]:


dmmB = dataB - meanB #calculating data minus mean
print('dmmB:')
print(dmmB)

covB = np.dot(np.transpose(dmmB),dmmB)/(len(dmmB)-1)
print('covB matrix:')
print(covB)
print('covB shape:', covB.shape)

#confirming the results with the np.cov function 
covB_2 = np.cov(dmmB.T)
print('covB_2 matrix:')
print(covB_2)
print('covB_2 shape:', covB_2.shape)

#Testing out if data without the mean subtracted makes a difference 
covB_2b = np.cov(dataB.T) 
print('covB_2b matrix:')
print(covB_2b)
print('covB_2b shape:', covB_2b.shape)

#comparing the two matricies to see if they are actually equal?
covB.all() == covB_2.all()

check1= np.array_equal(covB,covB_2)
print('comparing if the two covariance matricies are equal:', check1)
check2= np.allclose(covB,covB_2)
print('comparing if the two covariance matricies are close enough:', check2)


# 

# # Question 3
# #### The data generated is random and normally distributed with a mean for dataA, dataB and covariance for dataA and dataB given in meanA.xlsx, meanB.xlsx, covarianceA.xlsx and covarianceB.xlsx respectively. Briefly explain why your answers are different from the parameters used to generate the data

# ### Uploading & Checking Data

# In[13]:


#Data from the meanA.csv
mean_A = np.loadtxt('meanA.csv', delimiter=',') 
print("mean_A:")
print(mean_A)

#Calculated mean from DataA
mean = np.mean(dataA,axis=0)
print('mean of dataA:')
print(mean)

#comparison to see if they are the same
check= np.array_equal(mean_A,mean)
print('comparing mean_A and mean:', check)


#Data from the meanB.csv
mean_B = np.loadtxt('meanB.csv', delimiter=',') 
print("mean_B:")
print(mean_B)

#Calculated mean from DataB
meanB = np.mean(dataB,axis=0)
print('mean of dataB:')
print(meanB)

#comparison to see if they are the same
check= np.array_equal(mean_B,meanB)
print('comparing mean_A and mean:', check)


# ### Discussion
# 
# ##### If dataA and dataB were randomly generated from the mean and covariance matricies of meanA, covarianceA and meanB, covarianceB respectively. Then dataA and dataB represent a subset of the original population. As such the mean and covariance measured for dataA and dataB are approximations of the original underlying population from which the dataA and dataB samples were gathered. 
# 
# ##### With each time a random data set is generated from meanA, covarianceA (as well as meanB, covarianceB) the subset data will be only an approximation of the original population. Therefore the subset is likely to approximate the original data and the new mean and covariance is likely to fall within a few standard deviations of the original mean and covariance (given that the distribution of the original population is uniform). See image below for visual explanation.
# 
# ##### credit: https://psychology.illinoisstate.edu/jccutti/psych340/fall02/oldlecturefiles/prob.html

# In[14]:


Image("Capture.JPG")


# In[ ]:





# # Question 4
# #### From the Excel file document multinormal.xlsx, I have provided 10 examples of samples drawn from a two-dimensional normal distribution (all in one file). In the data, there are 20 columns of 1000 samples. Example 1 is columns 1 and 2, example 2 is columns 3 and 4, etc. For each of the examples, the mean is [0 0]T but the covariance matrix changes.
# 

# #### Upload and Check data

# In[3]:


MN = np.loadtxt('multNormal.csv', delimiter=',') 
print("MN:")
print(MN)
print(MN.shape)
type(MN)
    


# #### A. Plot the 2D points as a scatter plot for each example.

# In[9]:


np.random.seed(1234)

i=0
j=1
for i in range(0,20,2):
    print ('Example ', j)
    j += 1
    # this temp array was made to visualize correct elements are being plotted.
    tmp = np.array(MN[:,i:i+2]) #change the rows to 0:5 to better visualize scatterplot
    #print(tmp)
    plt.figure()
    plt.scatter(tmp[:,0], tmp[:,1])
    plt.xlabel('column 1')
    plt.ylabel('column 2')
    plt.xlim(-15,15)
    plt.ylim(-15,15)
    plt.show()
  


# #### B. Calculate 10 separate covariance matrices. 

# In[5]:


i1=0
j1=1
for i1 in range(0,20,2):
    print ('Example ', j1)
    j1 += 1
    tmp = np.array(MN[:,i1:i1+2])
    #print(tmp)
    mean = np.mean(tmp, axis=0)
    #print('mean: ')
    #print(mean)
    tmp_m_m = tmp - mean
    #print('tmp_m_m:')
    #print(tmp_m_m)
    cov = np.cov(tmp_m_m.T)
    print('covariance:')
    print(cov)


# #### C. Plot the covariance value (the off-diagonal of the covariance matrix) for each example. 

# In[6]:


i1=0
#j1=1
k=0
#t=(10,2)
tmp_cov = np.empty([10,2])
#print (tmp_co.shape)
#print(tmp_cov)
for i1 in range(0,20,2):
    #print ('Example ', j1)
    #j1 += 1
    tmp = np.array(MN[:,i1:i1+2])
    #print(tmp)
    mean = np.mean(tmp, axis=0)
    #print('mean: ')
    #print(mean)
    tmp_m_m = tmp - mean
    #print('tmp_m_m:')
    #print(tmp_m_m)
    cov = np.cov(tmp_m_m.T)
    #print('covariance:')
    #print(cov)
    tmp_cov[k,0] = cov[0,1]
    tmp_cov[k,1] = cov[0,1]
    #print(tmp_cov)
    k+=1

    
print(tmp_cov)
colour = np.random.rand(10)
plt.figure()
plt.scatter(tmp_cov[:,0], tmp_cov[:,1], c=colour)
plt.xlabel('cov')
plt.ylabel('cov')
plt.title ('Covariance vs. Covariance')
plt.show()


# #### D. Similarly, plot the variance of the first column and the variance of the second column for each example.

# In[7]:


i1=0
#j1=1
k=0
#t=(10,2)
tmp_var = np.empty([10,2])
#print (tmp_co.shape)
#print(tmp_cov)
for i1 in range(0,20,2):
    #print ('Example ', j1)
    #j1 += 1
    tmp = np.array(MN[:,i1:i1+2])
    #print(tmp)
    mean = np.mean(tmp, axis=0)
    #print('mean: ')
    #print(mean)
    tmp_m_m = tmp - mean
    #print('tmp_m_m:')
    #print(tmp_m_m)
    cov = np.cov(tmp_m_m.T)
    #print('covariance:')
    #print(cov)
    tmp_var[k,0] = cov[0,0]
    tmp_var[k,1] = cov[1,1]
    #print(tmp_cov)
    k+=1

print(tmp_var) # this is actually the variance matrix in this case
colour = np.random.rand(10)
plt.figure()
plt.scatter(tmp_var[:,0], tmp_var[:,1], c=colour)
plt.xlabel('Var 1')
plt.ylabel('Var 2')
plt.title ('Variance 1 vs. Variance 2')
plt.show()


# #### E. From the 2D scatter plots, what has changed from example to example? Think geometrically!

# ##### Answer
# 
# ##### For the variance 1 vs variance 2 plot, its easy to visualize through the graph and the matrix that as Var1 increases Var2 decreases. Variance is also the measure of how much the values are spread around the mean. As seen in the graphs in question 4 part A, its very obvious that there is an approximate 90 degree counter-clockwise rotation in the variance of points from example 1 to 10. This can also be observed in the matrix printed in part D, as the variance in the column 1 continues to decrease and simultaneously increase in column 2 of each example. 
# 
# ##### Covariance is a visualization of relationship and its directionality between the two columns. That's why is the same in either direction covariance(C1,C2) == covariance(C2,C1). A high positive covariance means that as X increases, Y increases and there is a strong positive relationship and predictability between the two variable. In example 1 and 10, the covariances are very low numbers, and negative, it suggests thate there is a weak and inverse relationship between the X and Y variables of the examples 1 and 10, which can be confirmed from graphs in part A.

# #### F. How might I have generated the covariance matrices that were used? Consider three things:
# 
#     1. What matrix transformation did I apply to change the covariance matrix for each example?
#     2. What properties must a covariance have? (This one is trickier – one property is the matrix must be symmetrical but there is one more important property…)
#     3. How might one go about creating a covariance matrix that accomplishes the transformation mentioned in point 1 and has the properties mentioned in point 2?

# ##### Answer 1.
# 
# ##### An ~9 degree counter-clockwise rotation in the correlation of the points from one example to the next would allow generating the observed covariance matrices.
# 
# ##### Answer 2.
# 
# ##### The diagonal entries on the covariance matrix represent the variance of the values used to generate the covariance matrix. I'm not sure which property you were asking for. Also, it must always have the lowest possible rank. The covariance matrix will depend on the maximum linearly independent row or column vectors in the matrix whichever one is the lowest dimension. 
# 
# ##### Answer 3.
# 
# ##### The off-diagonal values need to always be the same. The covariance values can start at an arbitrarily small value and increase to about example 5 and then start to decrease again (as seen in the tmp_cov output below). The two values of the diagonal move in opposite directions (as seen in the tmp_var output below).

# In[14]:


print('tmp_cov')
print(tmp_cov)
print('tmp_var')
print(tmp_var)


# In[ ]:





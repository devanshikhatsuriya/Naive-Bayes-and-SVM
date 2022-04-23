#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import cvxopt
import time

import sys


# In[2]:

if len(sys.argv) != 3:
	print("Please provide relative or absolute <path_of_train_data> and <path_of_test_data> as command line arguments.")
	sys.exit()
train_path = sys.argv[1]
test_path = sys.argv[2]

# train_df = pandas.read_csv('../mnist/train.csv', header=None)
# test_df = pandas.read_csv('../mnist/test.csv', header=None)
try:
	train_df = pandas.read_csv(train_path, header=None)
	test_df = pandas.read_csv(test_path, header=None)
except:
	print("Error: Incorrect path for data")
	sys.exit()


# In[2]:

# print(train_df.shape)
# print(test_df.shape)
# print(train_df)


# In[3]:


train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

m = train_data.shape[0] # no. of training examples
n = train_data.shape[1]-1 # no. of features, last column is label

train_X = train_data[:,0:n]
train_Y = train_data[:,n]
# print(train_X.shape)
# print(train_Y.shape)

test_X = test_data[:,0:n]
test_Y = test_data[:,n]
# print(test_X.shape)
# print(test_Y.shape)


# In[4]:


#print(train_X[0])
train_X = train_X/255 # rescale from [0,255] to [0,1]
test_X = test_X/255
#print(train_X[0])


# In[5]:


# extract examples with labels 4 and 5
print("Extracting examples with labels 4 and 5...")
mask = numpy.any([train_Y==4, train_Y==5], axis=0)
# print(mask)
# print(train_Y[mask])
train_X = train_X[mask]
train_Y = train_Y[mask]
# print(train_X.shape)
# print(train_Y.shape)

mask = numpy.any([test_Y==4, test_Y==5], axis=0)
# print(mask)
# print(test_Y[mask])
test_X = test_X[mask]
test_Y = test_Y[mask]
# print(test_X.shape)
# print(test_Y.shape)

train_Y = numpy.reshape(train_Y, newshape=(train_Y.size,1))
test_Y = numpy.reshape(test_Y, newshape=(test_Y.size,1))
# print(train_Y.shape)
# print(test_Y.shape)

m = train_X.shape[0] # new no. of training examples
# print(m)


# In[6]:


C = 1.0 # the constant multiplied with summation eplsilon i


# In[7]:


# label digit 4 as -1, digit 5 as 1
print("Encoding label 4 as -1 and 5 and 1...")
train_Y_labels = numpy.where(train_Y==4, -1, 1)
test_Y_labels = numpy.where(test_Y==4, -1, 1)
# print(train_Y_labels)


# In[8]:


XY = train_Y_labels * train_X # broadcasted elementwise product (m*1 , m*n --> m*n matrix)
# print(XY.shape)

P = numpy.matmul(XY, numpy.transpose(XY, axes=(1,0)))
# print(P.shape)


# In[9]:

print("Generating coefficient matrices for CVXOPT optimization...")

XY = train_Y_labels * train_X # broadcasted elementwise product (m*1 , m*n --> m*n matrix)

P = cvxopt.matrix( numpy.matmul(XY, numpy.transpose(XY, axes=(1,0))) , tc='d' )
q = cvxopt.matrix( (-1) * numpy.ones((m,)) , tc='d' )
G = cvxopt.matrix( numpy.concatenate((numpy.eye(m), -1*numpy.eye(m)), axis=0) , tc='d' )
h = cvxopt.matrix( numpy.concatenate((numpy.ones((m,)), numpy.zeros(m,))) , tc='d' )
A = cvxopt.matrix( numpy.transpose(train_Y_labels, axes=(1,0)) , tc='d' )
b = cvxopt.matrix( 0.0 , tc='d' )


# In[10]:


# print(A)


# In[11]:

print("Optimizing using CVXOPT...")
start_time = time.process_time()
sol = cvxopt.solvers.qp(P, q, G, h, A, b)
time_taken = time.process_time() - start_time
print("Time taken in finding optimal solution:", time_taken, "seconds")
# print(sol['x'])
# print(sol['primal objective'])


# In[12]:


alpha = numpy.array(sol['x'])
alpha[alpha<1e-4] = 0
alpha[alpha>0.95] = 1
print("Alpha:", alpha)


# In[13]:


# PRINT SUPPORT VECTORS
print("No. of support vectors, nSV:", numpy.count_nonzero(alpha))
SV = train_X[alpha.flatten()!=0]
print("Support Vectors:", SV)


# In[14]:


w = numpy.sum(alpha * train_Y_labels * train_X, axis=0) # broadcasted elementwise product (m*1, m*1 , m*n --> m*n matrix)
w = numpy.reshape(w, newshape=(w.size,1))
print("w:", w)


# In[25]:


wTx = numpy.matmul(train_X, w) # m*1 matrix, [i,0] th element is wTx(i)


# In[24]:


b_values = train_Y_labels[alpha.flatten()!=0] - wTx[alpha.flatten()!=0]
# print(b_values)
b = numpy.sum(b_values)/b_values.size # all b_values are approximately same, so take average
print("b:", b)


# In[21]:


test_prediction = numpy.where((numpy.matmul(test_X, w) + b) >= 1, 1, -1)
test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_Y_labels)/test_Y_labels.size
print("Test Set Accuracy:", test_accuracy, "%")

# store predictions
prediction_data = numpy.asarray(numpy.where(test_prediction==-1, 4, 5))
numpy.savetxt('2_1a_test_prediction.csv', prediction_data, delimiter=',')




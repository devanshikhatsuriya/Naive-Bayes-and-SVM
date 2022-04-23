#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:

print("Gaussian Kernel SVM")

# print(train_df.shape)
# print(test_df.shape)
# print(train_df)


# In[4]:


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


# In[5]:


#print(train_X[0])
train_X = train_X/255 # rescale from [0,255] to [0,1]
test_X = test_X/255
#print(train_X[0])


# In[6]:


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


# In[7]:


C = 1.0 # the constant multiplied with summation eplsilon i
gamma = 0.05 # used in kernel_fn

def kernel_fn(x, z):
    """
        the gaussian kernel function
        is vectorized to handle vectors (n*1) as well as matrices (each row being a vector, i.e. m*n)
    """
    if x.ndim == 1 or x.shape[1] == 1: # x and z are vectors
        norm = numpy.matmul((x-z).T, (x-z)) 
    elif z.ndim == 1 or z.shape[1] == 1: # x is matrix, z is vector
        norm = numpy.matmul((x-z.T).T, (x-z.T)) # z.T will be a 1*n vector, which gets broadcasted into m*n on subtraction
    else: # x and z are matrices (m1*n and m2*n respectively)
        self_dot_x = numpy.sum(x*x, axis=1).reshape((x.shape[0],1)) # j th element is dot(x[j,:], x[j,:])
        self_dot_z = numpy.sum(z*z, axis=1).reshape((z.shape[0],1))
        norm = self_dot_x - numpy.matmul(x, z.T) - numpy.matmul(z, x.T).T + self_dot_z.T # (m1*m2) matrix
    return numpy.exp(-1 * gamma * norm)    
    


# In[8]:


# label digit 4 as -1, digit 5 as 1
print("Encoding label 4 as -1 and 5 and 1...")
train_Y_labels = numpy.where(train_Y==4, -1, 1)
test_Y_labels = numpy.where(test_Y==4, -1, 1)
# print(train_Y_labels)


# In[9]:

print("Generating coefficient matrices for CVXOPT optimization...")


P = cvxopt.matrix( train_Y_labels * kernel_fn(train_X, train_X) * train_Y_labels.T, tc='d' ) # broadcasted elementwise product (m*1, m*m, m*1 --> m*m matrix)
q = cvxopt.matrix( (-1) * numpy.ones((m,)) , tc='d' )
G = cvxopt.matrix( numpy.concatenate((numpy.eye(m), -1*numpy.eye(m)), axis=0) , tc='d' )
h = cvxopt.matrix( numpy.concatenate((numpy.ones((m,)), numpy.zeros(m,))) , tc='d' )
A = cvxopt.matrix( numpy.transpose(train_Y_labels, axes=(1,0)) , tc='d' )
b = cvxopt.matrix( 0.0 , tc='d' )


# In[10]:

print("Optimizing using CVXOPT...")
start_time = time.process_time()
sol = cvxopt.solvers.qp(P, q, G, h, A, b)
time_taken = time.process_time() - start_time
print("Time taken in finding optimal solution:", time_taken, "seconds")
# print(sol['x'])
# print(sol['primal objective'])


# In[11]:

alpha = numpy.array(sol['x'])
alpha[alpha<1e-4] = 0
alpha[alpha>0.9999] = 1
print("Alpha:", alpha)


# In[120]:


# PRINT SUPPORT VECTORS
print("No. of support vectors, nSV:", numpy.count_nonzero(numpy.all([alpha!=0, alpha!=1], axis=0)))
SV = train_X[alpha.flatten()!=0]
print("Support Vectors:", SV)


# In[129]:


wT_phix = numpy.sum(((alpha * train_Y_labels) * kernel_fn(train_X, train_X)), axis=0).reshape((train_Y_labels.size,1)) # m*1 matrix, [i,0] th element is wT dot phi(x(i))
# print(wT_phix)

b_values = train_Y_labels[alpha.flatten()!=0] - numpy.sum(((alpha[alpha.flatten()!=0] * train_Y_labels[alpha.flatten()!=0]) * kernel_fn(SV, SV)), axis=0).reshape((train_Y_labels[alpha.flatten()!=0].shape))
# print("b_values:", b_values)

b = numpy.sum(b_values)/b_values.size # all b_values are approximately same, so take average
print("b:", b)


# In[127]:


train_prediction = numpy.where(numpy.sum(((alpha * train_Y_labels) * kernel_fn(train_X, train_X)), axis=0) + b >= 0, 1, -1).reshape((train_Y_labels.size,1))
train_accuracy = 100 * numpy.count_nonzero(train_prediction==train_Y_labels)/train_Y_labels.size
print("Training Set Accuracy:", train_accuracy, "%")


# In[128]:


test_prediction = numpy.where(numpy.sum(((alpha * train_Y_labels) * kernel_fn(train_X, test_X)), axis=0) + b >= 0, 1, -1).reshape((test_Y_labels.size,1))
test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_Y_labels)/test_Y_labels.size
print("Test Set Accuracy:", test_accuracy, "%")

# store predictions
prediction_data = numpy.asarray(numpy.asarray(numpy.where(test_prediction==-1, 4, 5)))
numpy.savetxt('2_1b_test_prediction.csv', prediction_data, delimiter=',')



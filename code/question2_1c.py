#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas
import numpy
from libsvm.svmutil import *
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

# print(train_df.shape)
# print(test_df.shape)
# print(train_df)


# In[7]:


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


# In[8]:


#print(train_X[0])
train_X = train_X/255 # rescale from [0,255] to [0,1]
test_X = test_X/255
#print(train_X[0])


# In[9]:


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


# In[10]:


C = 1.0 # the constant multiplied with summation eplsilon i


# In[11]:


# label digit 4 as -1, digit 5 as 1
print("Encoding label 4 as -1 and 5 and 1...")
train_Y_labels = numpy.where(train_Y==4, -1, 1)
test_Y_labels = numpy.where(test_Y==4, -1, 1)
# print(train_Y_labels)


# In[24]:


# Linear Kernel
print("Training libsvm Model using Linear Kernel...")
start_time = time.process_time()
linear_model = svm_train(train_Y_labels.flatten(), train_X, f'-s 0 -t 0 -c {C}')
time_taken = time.process_time() - start_time
print("Time taken in training Linear Kernel SVM:", time_taken, "seconds")
p_label_train, p_acc_train, p_val_train = svm_predict(train_Y_labels.flatten(), train_X, linear_model)
p_label, p_acc, p_val = svm_predict(test_Y_labels.flatten(), test_X, linear_model)
# print(p_acc_train)
# print(p_acc)

# svm_predict()          : predict testing data
# svm_load_model()       : load a LIBSVM model.
# svm_save_model()       : save model to a file.
# evaluations()          : evaluate prediction results.



# In[25]:


# Gaussian Kernel 
print("Training libsvm Model using Gaussian Kernel...")
start_time = time.process_time()
gaussian_model = svm_train(train_Y_labels.flatten(), train_X, f'-s 0 -t 2 -g 0.05 -c {C}')
time_taken = time.process_time() - start_time
print("Time taken in training Gaussian Kernel SVM:", time_taken, "seconds")
p_label_train, p_acc_train, p_val_train = svm_predict(train_Y_labels.flatten(), train_X, gaussian_model)
p_label, p_acc, p_val = svm_predict(test_Y_labels.flatten(), test_X, gaussian_model)
# print(p_acc_train)
# print(p_acc)


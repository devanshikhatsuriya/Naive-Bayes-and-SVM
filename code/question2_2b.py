#!/usr/bin/env python
# coding: utf-8

# In[40]:


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

# In[42]:


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


# In[43]:


train_X = train_X/255 # rescale from [0,255] to [0,1]
test_X = test_X/255


# In[44]:


num_classes = numpy.unique(train_Y).size
num_classifiers = numpy.int64(((num_classes)*(num_classes-1))/2)
# print(num_classes, num_classifiers)   


# In[30]:

print(f"Training {num_classifiers} Models...")

models = [None for i in range(num_classifiers)]

classifier_num = 0

for class1 in range(num_classes):
    
    for class2 in range(class1+1, num_classes):
        
        print("\tClassifier between classes", class1, "and", class2, "...")
        
        # extract examples with labels class1 and class2
        
        mask = numpy.any([train_Y==class1, train_Y==class2], axis=0)
        train_X_subset = train_X[mask]
        train_Y_subset = train_Y[mask]

        mask = numpy.any([test_Y==class1, test_Y==class2], axis=0)
        test_X_subset = test_X[mask]
        test_Y_subset = test_Y[mask]

        train_Y_subset = numpy.reshape(train_Y_subset, newshape=(train_Y_subset.size,1))
        test_Y_subset = numpy.reshape(test_Y_subset, newshape=(test_Y_subset.size,1))

        m_new = train_X_subset.shape[0] # new no. of training examples
        
        # encode labels as -1 or 1
        
        train_Y_labels = numpy.where(train_Y_subset==class1, -1, 1)
        test_Y_labels = numpy.where(test_Y_subset==class1, -1, 1)
        
        # solve using optimizer

        start_time = time.process_time()
        models[classifier_num] = svm_train(train_Y_labels.flatten(), train_X_subset, '-s 0 -t 2 -g 0.05 -c 1')
        time_taken = time.process_time() - start_time
        print("\tTime taken in training model:", time_taken, "seconds")

        classifier_num += 1


# In[ ]:


def multi_class_prediction(test_X, test_Y, model_set, num_classes):
    
    m_test = test_X.shape[0]
    num_classifiers = len(model_set)
    vote_values = numpy.zeros((m_test, num_classes))
    score_values = numpy.zeros((m_test, num_classes)) # to break ties
    
    classifier_num = 0

    for class1 in range(num_classes):
        
        for class2 in range(class1+1, num_classes):  
            
            # encode labels as -1 or 1  
            test_Y_labels = numpy.where(test_Y==class1, -1, 1)  
            
            # prediction
            print("Prediction using Classifier between classes", class1, "and", class2, ".Log [Please Ignore]: ", end="")
            p_label, _ , p_val = svm_predict(test_Y_labels.flatten(), test_X, model_set[classifier_num]) 
            
            prediction_labels = numpy.array(p_label)
            prediction = numpy.where(prediction_labels==-1, class1, class2)
            
            votes = numpy.zeros((m_test, num_classes), dtype=numpy.int64)
            # print(votes.shape, numpy.arange(m_test).shape, prediction.flatten().shape)
            votes[numpy.arange(m_test), prediction.flatten()] = 1
            
            scores = numpy.zeros((m_test, num_classes), dtype=numpy.float64)
            scores[numpy.arange(m_test), prediction.flatten()] = abs(numpy.array(p_val)).flatten()
            
            vote_values += votes 
            score_values += scores
            
            classifier_num += 1
            
    max_votes = numpy.amax(vote_values,axis=1).reshape(vote_values.shape[0],1)
    mask = (vote_values!=max_votes)
    score_values[mask] = 0
    final_prediction = numpy.argmax(score_values, axis=1)
    
    return final_prediction


# In[ ]:

test_prediction = multi_class_prediction(test_X, test_Y, models, num_classes)
test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_Y)/test_Y.size
print("Test Set Accuracy:", test_accuracy, "%")

# store predictions
prediction_data = numpy.asarray(test_prediction)
numpy.savetxt('2_2b_test_prediction.csv', prediction_data, delimiter=',')


# In[ ]:


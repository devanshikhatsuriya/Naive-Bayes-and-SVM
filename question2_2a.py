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


train_X = train_X/255 # rescale from [0,255] to [0,1]
test_X = test_X/255


# In[5]:


num_classes = numpy.unique(train_Y).size
num_classifiers = numpy.int64(((num_classes)*(num_classes-1))/2)
# print(num_classes, num_classifiers)


# In[6]:


# for each classifier, we store a m*1 column of alpha-values
# use the fact that no. of examples of each class are m/num_classes, so no. of alpha for each model will be 2*m/num_classes
alpha_parameters = numpy.zeros(shape=(2*m//num_classes, num_classifiers), dtype=numpy.float64)


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
    


# In[ ]:
'''
# TO REGENERATE '2_2a_alpha_values.csv', REMOVE QUOTES OF THIS COMMENT BLOCK AND RE-RUN
# NOTE: If regenerating, IT IS REQUIRED TO LET THE COMPLETE CODE TO RUN to regenerate alpha values, otherwise there will be error in predictions
# NOTE: IT MAY TAKE UPTO 2 HOURS TO RUN

classifier_num = 0

for class1 in range(num_classes):
    
    for class2 in range(class1+1, num_classes):
        
        print("Classifier between classes", class1, "and", class2, "...")
        
        # extract examples with labels class1 and class2
        
        print("\tExtracting examples...")
        
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
        
        # generate coefficient matrices
        
        print("\tGenerating Coefficient Matrices...")
        
        P = cvxopt.matrix( train_Y_labels * kernel_fn(train_X_subset, train_X_subset) * train_Y_labels.T, tc='d' ) # broadcasted elementwise product (m_new*1, m_new*m_new, 1*m_new --> m_new*m_new matrix)
        q = cvxopt.matrix( (-1) * numpy.ones((m_new,)) , tc='d' )
        G = cvxopt.matrix( numpy.concatenate((numpy.eye(m_new), -1*numpy.eye(m_new)), axis=0) , tc='d' )
        h = cvxopt.matrix( numpy.concatenate((numpy.ones((m_new,)), numpy.zeros(m_new,))) , tc='d' )
        A = cvxopt.matrix( numpy.transpose(train_Y_labels, axes=(1,0)) , tc='d' )
        b = cvxopt.matrix( 0.0 , tc='d' )
        
        # solve using optimizer
        
        print("\tFinding Optimal Solution...")
        
        start_time = time.process_time()
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        time_taken = time.process_time() - start_time
        print("\tTime taken in finding Optimal Parameters:", time_taken, "seconds")
        # print(sol['x'])
        # print("\tPrimal Objective Value:", -1*sol['primal objective'])
        
        # store alpha value
        
        alpha = numpy.array(sol['x'])
        alpha_parameters[:,classifier_num] = alpha.flatten()        
        classifier_num += 1        


# In[ ]:


# store alpha parameters
print("Saving alpha values...")
alpha_data = numpy.asarray(alpha_parameters)
numpy.savetxt('2_2a_alpha_values.csv', alpha_data, delimiter=',')
'''

# In[5]:


# alpha_data = numpy.load('alpha_values.npy')
print("Using cached alpha values...")
print("[NOTE: To regenerate '2_2a_alpha_values.csv', please see NOTE in code]")
alpha_data = numpy.loadtxt('2_2a_alpha_values.csv', delimiter=',')
print(alpha_data)

alpha_parameters = alpha_data


# In[ ]:


def multi_class_prediction(test_X, alpha_parameters, train_X, train_Y, num_classes):
    
    m_test = test_X.shape[0]
    num_classifiers = alpha_parameters.shape[1]
    vote_values = numpy.zeros((m_test, num_classes))
    score_values = numpy.zeros((m_test, num_classes)) # to break ties
    
    classifier_num = 0

    for class1 in range(num_classes):
        
        for class2 in range(class1+1, num_classes):

            print("\tUsing classifier between", class1, "and", class2, "...")  
            
            # extract train examples with labels class1 and class2 for parameters
            mask = numpy.any([train_Y==class1, train_Y==class2], axis=0)
            train_X_subset = train_X[mask]
            train_Y_subset = train_Y[mask]
            train_Y_subset = numpy.reshape(train_Y_subset, newshape=(train_Y_subset.size,1))
            m_new = train_X_subset.shape[0] # new no. of training examples
            # encode labels as -1 or 1
            train_Y_labels = numpy.where(train_Y_subset==class1, -1, 1)  
            
            # prediction            
            alpha = alpha_parameters[:,classifier_num]
            alpha = numpy.reshape(alpha, newshape=(alpha.size,1))
            alpha[alpha<1e-5] = 0
            alpha[alpha>1-(1e-4)] = 1.0
            wT_xTest = numpy.sum((alpha * train_Y_labels * kernel_fn(train_X_subset, test_X)), axis=0).reshape((m_test,1))
            # SV_mask = numpy.all([alpha.flatten()!=0, alpha.flatten()!=C], axis=1)
            SV_mask = (alpha.flatten()!=0)
            SV = train_X_subset[SV_mask]
            wT_SV = numpy.sum(((alpha[SV_mask] * train_Y_labels[SV_mask]) * kernel_fn(SV, SV)), axis=0).reshape((train_Y_labels[SV_mask].shape))
            b_values = train_Y_labels[SV_mask] - wT_SV 
            # print(b_values)
            b = numpy.sum(b_values)/b_values.size # all values are approximately equal, so take average
            # print(b)
            prediction_labels = numpy.where(wT_xTest + b >= 0, 1, -1)
            
            prediction = numpy.where(prediction_labels==-1, class1, class2)
            
            votes = numpy.zeros((m_test, num_classes), dtype=numpy.int64)
            # print(votes.shape, numpy.arange(m_test).shape, prediction.flatten().shape)
            votes[numpy.arange(m_test), prediction.flatten()] = 1
            
            scores = numpy.zeros((m_test, num_classes), dtype=numpy.float64)
            scores[numpy.arange(m_test), prediction.flatten()] = abs(wT_xTest + b).flatten()
            
            vote_values += votes 
            score_values += scores
            
            classifier_num += 1
            
    max_votes = numpy.amax(vote_values,axis=1).reshape(vote_values.shape[0],1)
    mask = (vote_values!=max_votes)
    score_values[mask] = 0
    final_prediction = numpy.argmax(score_values, axis=1)
    
    return final_prediction


# In[ ]:

# train_prediction = multi_class_prediction(train_X, alpha_parameters, train_X, train_Y, num_classes)
# train_accuracy = 100 * numpy.count_nonzero(train_prediction==train_Y)/train_Y.size
# print("Train Set Accuracy:", train_accuracy, "%")

print("Making predictions on Test Set...")
test_prediction = multi_class_prediction(test_X, alpha_parameters, train_X, train_Y, num_classes)
test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_Y)/test_Y.size
print("Test Set Accuracy:", test_accuracy, "%")

# store predictions
prediction_data = numpy.asarray(test_prediction)
numpy.savetxt('2_2a_test_prediction.csv', prediction_data, delimiter=',')


# In[ ]:




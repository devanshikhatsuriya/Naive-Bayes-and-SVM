
import pandas
import numpy
from libsvm.svmutil import *
import time
import matplotlib.pyplot as plt

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

C_values = numpy.array([1e-5, 1e-3, 1, 5, 10])
K = 5

train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

m = train_data.shape[0] # no. of training examples
n = train_data.shape[1]-1 # no. of features, last column is label

train_X_all = train_data[:,0:n]
train_Y_all = train_data[:,n]

test_X = test_data[:,0:n]
test_Y = test_data[:,n]

train_X_all = train_X_all/255 # rescale from [0,255] to [0,1]
test_X = test_X/255

num_classes = numpy.unique(train_Y_all).size
num_classifiers = numpy.int64(((num_classes)*(num_classes-1))/2)

numpy.random.seed(42)

perm = numpy.random.permutation(m)
train_X_all = train_X_all[perm]
train_Y_all = train_Y_all[perm]

fold_size = m//K

avg_cross_validation_accuracy = numpy.zeros((C_values.size,))
best_cross_validation_accuracy = numpy.zeros((C_values.size,))
test_accuracies = numpy.zeros((C_values.size,))

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
            print("\t\t\tPrediction using Classifier between classes", class1, "and", class2, ".Log [Please Ignore]: ", end="")
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
    
'''
# TO REGENERATE '2_2d_accuracies.csv', DELETE '2_2d_accuracies.csv' and REMOVE QUOTES OF THIS COMMENT BLOCK AND RE-RUN
# NOTE: IT MAY TAKE UPTO 5 HOURS TO RUN

index = 0

for C in C_values:

    print("For C value:", C)

    best_accuracy = None
    best_model_set = None

    validation_accuracies = numpy.zeros((K,))

    for validation_set_no in range(K):

        print(f"\tValidation Set No. {validation_set_no}...")

        start_ind = validation_set_no*fold_size
        end_ind = validation_set_no*fold_size + fold_size
        validation_X = train_X_all[start_ind:end_ind, :] 
        validation_Y = train_Y_all[start_ind:end_ind]
        
        train_X = numpy.concatenate((train_X_all[:start_ind], train_X_all[end_ind:]), axis=0)
        train_Y = numpy.concatenate((train_Y_all[:start_ind], train_Y_all[end_ind:]))

        models = [None for i in range(num_classifiers)]

        print(f"\t\tTraining {num_classifiers} Models...")

        classifier_num = 0

        for class1 in range(num_classes):
            
            for class2 in range(class1+1, num_classes):
                
                print("\t\t\tClassifier between classes", class1, "and", class2, "...")
                
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
                models[classifier_num] = svm_train(train_Y_labels.flatten(), train_X_subset, f"-s 0 -t 2 -g 0.05 -c {C} -q")
                time_taken = time.process_time() - start_time
                print("\t\t\tTime taken in finding Optimal Parameters:", time_taken, "seconds")

                classifier_num += 1

        # compute validation accuracy

        print(f"\t\tPredicting using {num_classifiers} Models...")

        validation_prediction = multi_class_prediction(validation_X, validation_Y, models, num_classes)
        validation_accuracy = 100 * numpy.count_nonzero(validation_prediction==validation_Y)/validation_Y.size
        print("\t\tValidation Set Accuracy:", validation_accuracy, "%")

        validation_accuracies[validation_set_no] = validation_accuracy

        # update best model set so far for this C value

        if (best_accuracy == None) or (best_accuracy < validation_accuracy):
            best_accuracy = validation_accuracy
            best_model_set = models

    # compute test accuracy using best cross validation model

    print("\tComputing Test Set Accuracy using Best Cross Validation Model")

    test_prediction = multi_class_prediction(test_X, test_Y, best_model_set, num_classes)
    test_accuracy = 100 * numpy.count_nonzero(test_prediction==test_Y)/test_Y.size
    print("\tTest Set Accuracy:", test_accuracy, "%")

    avg_cross_validation_accuracy[index] = numpy.mean(validation_accuracies)
    best_cross_validation_accuracy[index] = numpy.amax(validation_accuracies)
    test_accuracies[index] = test_accuracy

    # store values
    print("\tStoring accuracy values...")
    with open('2_2d_accuracies.csv', 'a') as file:
        file.write(f"{C},{avg_cross_validation_accuracy[index]},{best_cross_validation_accuracy[index]},{test_accuracies[index]}\n")

    index += 1
'''

# plot

print("Using cached accuracy values from file '2_2d_accuracies.csv'...")
print("[NOTE: To regenerate '2_2d_accuracies.csv', see NOTE in code]")
print("Plotting accuracies...")

with open('2_2d_accuracies.csv', 'r') as f:
    lines = f.read().split('\n')
    for i in range(5):
        acc = lines[i].split(',')
        avg_cross_validation_accuracy[i] = acc[1]
        best_cross_validation_accuracy[i] = acc[2]
        test_accuracies[i] = acc[3]

fig = plt.figure(figsize=(8, 8))

plt.plot(numpy.log(C_values), avg_cross_validation_accuracy, label="Avg. C.V. Accuracy", linestyle='solid')
plt.plot(numpy.log(C_values), best_cross_validation_accuracy, label="Best C.V. Accuracy", linestyle='solid')
plt.plot(numpy.log(C_values), test_accuracies, label="Test Accuracy", linestyle='solid')

plt.scatter(numpy.log(C_values), avg_cross_validation_accuracy, label="Avg. C.V. Accuracy")
plt.scatter(numpy.log(C_values), best_cross_validation_accuracy, label="Best C.V. Accuracy")
plt.scatter(numpy.log(C_values), test_accuracies, label="Test Accuracy")

plt.xlabel("log(C)")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("2_2d_cross_validation.png", dpi=100)
plt.show()









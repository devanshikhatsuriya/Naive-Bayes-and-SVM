
import pandas
import numpy

import sys


# In[2]:

if len(sys.argv) != 3:
	print("Please provide relative or absolute <path_of_train_data> and <path_of_test_data> as command line arguments.")
	sys.exit()
train_path = sys.argv[1]
test_path = sys.argv[2]

# test_data_df = pandas.read_json('../Music_reviews_json/reviews_Digital_Music_5.json/Music_Review_test.json', lines=True)
try:
	test_data_df = pandas.read_json(test_path, lines=True)
except:
	print("Error: Incorrect path for data")
	sys.exit()

test_labels = test_data_df["overall"].to_numpy()

# Test set accuracy by Random Classifier
print("\nRandom Classifier...")
random_prediction = numpy.random.randint(low=1, high=6, size=(test_labels.size,))
# print(random_prediction)
# print(numpy.count_nonzero(random_prediction==0), numpy.count_nonzero(random_prediction==1), numpy.count_nonzero(random_prediction==2), numpy.count_nonzero(random_prediction==3) , numpy.count_nonzero(random_prediction==4), numpy.count_nonzero(random_prediction==5), numpy.count_nonzero(random_prediction==6))
random_accuracy = 100* numpy.count_nonzero(test_labels==random_prediction)/test_labels.size
print("Test Accuracy by Random Classifier:", random_accuracy,"%")

# Test set accuracy by Majority Classifier
print("\nMajority Classifier...")
counts = numpy.bincount(test_labels)
majority_class = numpy.argmax(counts)
print("Majority Class:", majority_class)
majority_accuracy = 100* numpy.count_nonzero(test_labels==majority_class)/test_labels.size
print("Test Accuracy by Majority Classifier:", majority_accuracy,"%\n")

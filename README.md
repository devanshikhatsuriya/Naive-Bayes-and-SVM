# Stochastic-Gradient-Descent-and-GDA
## Assignment 2, COL774 Machine Learning, Sem I, 2021-22
This assignment has 2 parts: 
1. Text Classification using Naive Bayes
2. MNIST Digit Classification using SVMs

## Running

The bash script code/run.sh can be used to run the code files. The first input argument is always the question number, the second argument is relative or absolute path of the train file, the third argument is absolute or relative path of test file and further arguments, if any, depend on the question.
#### Arguments for different questions:

##### Question 1
	./run.sh 1 <path_of_train_data> <path_of_test_data> <part_num>
Here, 'part_num' can be a-g.
NOTE: This format is different from specified format. 'part-num' can be f.
	
##### Question 2
    ./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num>
Here, 'binary_or_multi_class' is 0 for binary classification and 1 for multi-class and 'part_num' is part number which can be a-c for binary classification and a-d for multi-class.


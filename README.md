

The commands to run different parts of different questions are as follows:

	The first input argument is always the question number, second argument - relative or absolute path of the train file, third argument - absolute or relative path of test file and further arguments, if any depends on the question.

	Arguments for different questions:

	Question 1
		./run.sh 1 <path_of_train_data> <path_of_test_data> <part_num>

	Here, 'part_num' can be a-g.
	
	 
	NOTE: This format is different from specified format. 'part-num' can be f.
	
	

	Question 2
		./run.sh 2 <path_of_train_data> <path_of_test_data> <binary_or_multi_class> <part_num>

	Here, 'binary_or_multi_class' is 0 for binary classification and 1 for multi-class. 
	'part_num' is part number which can be a-c for binary classification and a-d for multi-class.
	
	
	
Package Versions Used:
	1) Python 3.8.10
	2) cvxopt==1.2.7
	3) libsvm-official==3.25.0
	4) matplotlib==3.4.3
	5) matplotlib-inline==0.1.2
	6) nltk==3.6.3
	7) numpy==1.21.2
	8) pandas==1.3.2


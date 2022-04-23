#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas
import string
import numpy
import time

import sys


# In[2]:

if len(sys.argv) != 3:
	print("Please provide relative or absolute <path_of_train_data> and <path_of_test_data> as command line arguments.")
	sys.exit()
train_path = sys.argv[1]
test_path = sys.argv[2]

# train_data_df = pandas.read_json('../Music_reviews_json/reviews_Digital_Music_5.json/Music_Review_train.json', lines=True)
# test_data_df = pandas.read_json('../Music_reviews_json/reviews_Digital_Music_5.json/Music_Review_test.json', lines=True)
try:
	train_data_df = pandas.read_json(train_path, lines=True)
	test_data_df = pandas.read_json(test_path, lines=True)
except:
	print("Error: Incorrect path for data")
	sys.exit()
	
nltk.download()


# In[3]:


def lower_and_remove_punctuation(tokenized_review):
    return list(map(lambda token: "".join([c for c in token.lower() if c not in string.punctuation]), tokenized_review))

def remove_non_alphabetic(tokenized_review):
    return [w for w in tokenized_review if w.isalpha()]


# In[4]:


# Split into tokens.
print("Tokenizing all reviews...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("Converting to lowercase and removing punctuation...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("Filtering out non-alphabetic tokens...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: remove_non_alphabetic(x))


# Further Processing: Removing Stop Words and Stemming

def filter_stop_words(tokenized_review, stop_words):
    return [w for w in tokenized_review if w not in stop_words]

def stem_words(tokenized_review, stemmer):
    return [stemmer.stem(w) for w in tokenized_review]

# print(train_data_df["reviewText"][0])

# process train and test data

# Filter out tokens that are stop words.
print("Filtering out stop-words...")
stop_words = set(nltk.corpus.stopwords.words('english'))
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: filter_stop_words(x, stop_words))
# Stem tokens.
print("Stemming all tokens...")
stemmer = nltk.stem.porter.PorterStemmer()
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: stem_words(x, stemmer))


# In[6]:


# print(numpy.array(train_reviewText[0]))


# In[7]:


'''
def compute_word_freq_vectorized(data_reviewText):
    """
        NOTE: this method is slower than below non-vectorized method
        input -> numpy array of lists containing tokens of sentences
        output -> dictionary containing word: freq assignments
                note that we use the same dictionary to find a word: id assignment by list(dict.keys()) method
    """
    vocab = {}
    def count(word):
        if word in vocab:
            vocab[word]+=1
        else:
            vocab[word]=1
    vcount = numpy.vectorize(count)
    f = lambda l: vcount(numpy.array(l))
    vf = numpy.vectorize(f)
    vf(data_reviewText)
    return vocab

# start = time.process_time()   

total_word_freqs = compute_word_freq_vectorized(train_reviewText)
print(len(total_word_freqs))
print(total_word_freqs)

# print(time.process_time() - start)
'''


# In[8]:


def compute_word_freq(data_reviewText):
    """
        input -> numpy array of lists containing tokens of sentences
        output -> dictionary containing word: freq assignments
        NOTE: We use the freq. dictionary of training data to find a word: id assignment using the method list(dict.keys())
        # this method is faster than the above vectorized method
    """
    vocab = {}
    for l in data_reviewText:
        np_l = numpy.array(l)
        for word in l:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
    return vocab


# In[9]:

# Retrain on Further Processed Test Data

## Build new vocabulary

new_train_reviewText = train_data_df["reviewText"].to_numpy(dtype=object)
train_labels = train_data_df["overall"].to_numpy()

print("Building vocabulary...")
start = time.process_time()
new_total_word_freqs = compute_word_freq(new_train_reviewText)
new_total_word_freqs["UNK"] = 1 # to account for unseen words in test data. 
                # Note that we have lower-cased all other words so no known word will be "UNK"
print(f"\tVocabulary built in {time.process_time() - start} seconds")

new_words = numpy.array(list(new_total_word_freqs.keys()))
new_vocab = {word: index for index, word in enumerate(new_words)} # the word: index dictionary
print("\tSize of new vocabulary:", len(new_vocab))

## Compute new parameters

print("Computing parameters...")
alpha = 1 # parameter which controls the strength of the smoothing
m = train_labels.size
V = len(new_vocab) # no. of words in vocab. 
            # Note that vocab includes the "UNK" word also, so V is actual no. of words in training data + 1
num_classes = 5
new_phi_y = numpy.empty((num_classes,))
print("\tComputing phi y=i...")
for i in range(5):
    new_phi_y[i] = numpy.count_nonzero(train_labels==(i+1))/m # same as phi_y
    
new_phi_k_y = numpy.zeros((5,len(new_vocab)))
print("\tComputing phi x_j=k|y=i (for any j)...")
for i in range(5):
    mask = (train_labels==(i+1))
    reviews = new_train_reviewText[mask] # extract examples of class (i+1)
    v_len = numpy.vectorize(lambda l: len(l))
    review_lens = v_len(reviews)
    total_words = numpy.sum(review_lens)
    print(f"\t\tComputing word frequencies for class {i+1}...")
    class_word_freqs = compute_word_freq(reviews)
    for word in numpy.array(list(new_total_word_freqs.keys())):
        if word not in class_word_freqs:
            freq_in_class = 0
        else:
             freq_in_class = class_word_freqs[word]      
        word_index = new_vocab[word]
        new_phi_k_y[i, word_index] = (freq_in_class+alpha)/(total_words+(V*alpha))

print("Parameters after stemming and stop-word removal:")
print("Phi y=i:", new_phi_y)
print("Phi x_j=k|y=i, for all j:\n", new_phi_k_y)


# In[11]:


def prediction(textReview, phi_y, phi_k_y, vocab):
    """
        Returns predicted class of textReview out of 1 to num_classes according to parameters phi_y and phi_k_y
            textReview: tokenized/processed input review
    """
    num_classes = phi_y.shape[0]
    log_probs = numpy.zeros(num_classes) 
    # log_probs will store log of un-normalized probabilities for all classes
    # i.e. P(y=i|x) proportional to P(x|y=i)*P(y=i)
    for i in range(num_classes):
        log_probs[i] += numpy.log(phi_y[i]) # log(P(y=i))
        for word in textReview:
            if word not in vocab:
                word_index = vocab["UNK"] # accounts for unseen words in test data
            else:
                word_index = vocab[word]
            word_prob_given_class = phi_k_y[i, word_index]
            log_probs[i] += numpy.log(word_prob_given_class) # sum of log(P(x_j|y=i)) for j=0 to length of input textReview
        
    return numpy.argmax(log_probs)+1        

vectorized_prediction = numpy.vectorize(prediction)


# In[12]:


# Process Test Data
print("Processing Test Data...")

# Split into tokens.
print("\tTokenizing test reviews...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("\tConverting to lowercase and removing punctuation...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("\tFiltering out non-alphabetic tokens...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: remove_non_alphabetic(x))
# Filter out tokens that are stop words.
print("\tFiltering out stop-words...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: filter_stop_words(x, stop_words))
# Stem tokens.
print("\tStemming all tokens...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: stem_words(x, stemmer))


# In[32]:


new_test_reviewText = test_data_df["reviewText"].to_numpy(dtype=object)
test_labels = test_data_df["overall"].to_numpy()
new_test_prediction = numpy.empty(test_labels.shape, test_labels.dtype)
print("Computing Test Predictions...")
start = time.process_time()
for i in range(new_test_prediction.size):
    new_test_prediction[i] = prediction(new_test_reviewText[i], new_phi_y, new_phi_k_y, new_vocab)
print(f"\tTest predictions computed in {time.process_time() - start} seconds")
    
new_train_prediction = numpy.empty(train_labels.shape, train_labels.dtype)
print("Computing Train Predictions...")
start = time.process_time()
for i in range(new_train_prediction.size):
    new_train_prediction[i] = prediction(new_train_reviewText[i], new_phi_y, new_phi_k_y, new_vocab)
print(f"\tTrain predictions computed in {time.process_time() - start} seconds")
    
# print(new_test_prediction) 
# print(new_train_prediction)

new_train_accuracy = 100* numpy.count_nonzero(train_labels==new_train_prediction)/train_labels.size
print("Train Set Accuracy by New Model:", new_train_accuracy,"%")
new_test_accuracy = 100* numpy.count_nonzero(test_labels==new_test_prediction)/test_labels.size
print("Test Accuracy by New Model:", new_test_accuracy,"%")


# store predictions
prediction_data = numpy.asarray(new_test_prediction)
numpy.savetxt('1_d_test_prediction.csv', prediction_data, delimiter=',')




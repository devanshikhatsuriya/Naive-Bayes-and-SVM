
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

# print(train_data_df["summary"])

def lower_and_remove_punctuation(tokenized):
    return list(map(lambda token: "".join([c for c in token.lower() if c not in string.punctuation]), tokenized))

def remove_non_alphabetic(tokenized):
    return [w for w in tokenized if w.isalpha()]
    
def filter_stop_words(tokenized, stop_words):
    return [w for w in tokenized if w not in stop_words]

def stem_words(tokenized, stemmer):
    return [stemmer.stem(w) for w in tokenized]

print("Tokenizing all reviews...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("Converting to lowercase and removing punctuation...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("Filtering out non-alphabetic tokens...")
train_data_df["reviewText"] = train_data_df["reviewText"].apply(lambda x: remove_non_alphabetic(x))
   
print("Tokenizing all summaries...")
train_data_df["summary"] = train_data_df["summary"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("Converting to lowercase and removing punctuation...")
train_data_df["summary"] = train_data_df["summary"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("Filtering out non-alphabetic tokens...")
train_data_df["summary"] = train_data_df["summary"].apply(lambda x: remove_non_alphabetic(x))

train_rT = train_data_df["reviewText"].to_numpy(dtype=object)
train_s = train_data_df["summary"].to_numpy(dtype=object)
print("Combining review and summary tokens...")
train_summary = numpy.empty((train_rT.shape[0]), dtype=object)
for i in range(len(train_summary)):
    train_summary[i] = train_rT[i] + train_s[i]
    
train_labels = train_data_df["overall"].to_numpy()



def compute_word_freq(data_tokens):
    """
        input -> numpy array of lists containing tokens of sentences
        output -> dictionary containing word: freq assignments
        NOTE: We use the freq. dictionary of training data to find a word: id assignment using the method list(dict.keys())
        # this method is faster than the above vectorized method
    """
    vocab = {}
    for l in data_tokens:
        np_l = numpy.array(l)
        for word in l:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
    return vocab
    

print("Building vocabulary...")
start = time.process_time()
total_word_freqs = compute_word_freq(train_summary)
total_word_freqs["UNK"] = 1 # to account for unseen words in test data. 
                # Note that we have lower-cased all other words so no known word will be "UNK"
print(f"\tVocabulary built in {time.process_time() - start} seconds")

words = numpy.array(list(total_word_freqs.keys()))
vocab = {word: index for index, word in enumerate(words)} # the word: index dictionary
print("\tSize of vocabulary:", len(vocab))

## Compute parameters

print("Computing parameters...")
alpha = 1 # parameter which controls the strength of the smoothing
m = train_labels.size
V = len(vocab) # no. of words in vocab. 
            # Note that vocab includes the "UNK" word also, so V is actual no. of words in training data + 1
num_classes = 5
phi_y = numpy.empty((num_classes,))
print("\tComputing phi y=i...")
for i in range(5):
    phi_y[i] = numpy.count_nonzero(train_labels==(i+1))/m # same as phi_y
    
phi_k_y = numpy.zeros((5,len(vocab)))
print("\tComputing phi x_j=k|y=i (for any j)...")
for i in range(5):
    mask = (train_labels==(i+1))
    summaries = train_summary[mask] # extract examples of class (i+1)
    v_len = numpy.vectorize(lambda l: len(l))
    summary_lens = v_len(summaries)
    total_words = numpy.sum(summary_lens)
    print(f"\t\tComputing word frequencies for class {i+1}...")
    class_word_freqs = compute_word_freq(summaries)
    for word in numpy.array(list(total_word_freqs.keys())):
        if word not in class_word_freqs:
            freq_in_class = 0
        else:
             freq_in_class = class_word_freqs[word]      
        word_index = vocab[word]
        phi_k_y[i, word_index] = (freq_in_class+alpha)/(total_words+(V*alpha))

print("Parameters on using Summary for Classification:")
print("Phi y=i:", phi_y)
print("Phi x_j=k|y=i, for all j:\n", phi_k_y)


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

print("\tTokenizing all reviews...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("\tConverting to lowercase and removing punctuation...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("\tFiltering out non-alphabetic tokens...")
test_data_df["reviewText"] = test_data_df["reviewText"].apply(lambda x: remove_non_alphabetic(x))

# Split into tokens.
print("\tTokenizing test summaries...")
test_data_df["summary"] = test_data_df["summary"].apply(lambda x: nltk.tokenize.word_tokenize(x))
# Convert to lowercase and remove punctuation from each token.
print("\tConverting to lowercase and removing punctuation...")
test_data_df["summary"] = test_data_df["summary"].apply(lambda x: lower_and_remove_punctuation(x))
# Filter out remaining tokens that are not alphabetic.
print("\tFiltering out non-alphabetic tokens...")
test_data_df["summary"] = test_data_df["summary"].apply(lambda x: remove_non_alphabetic(x))

test_rT = test_data_df["reviewText"].to_numpy(dtype=object)
test_s = test_data_df["summary"].to_numpy(dtype=object)
print("\tCombining review and summary tokens...")
test_summary = numpy.empty((test_rT.shape[0]), dtype=object)
for i in range(len(test_summary)):
    test_summary[i] = test_rT[i] + test_s[i]


# In[32]:


test_labels = test_data_df["overall"].to_numpy()
test_prediction = numpy.empty(test_labels.shape, test_labels.dtype)
print("Computing Test Predictions...")
start = time.process_time()
for i in range(test_prediction.size):
    test_prediction[i] = prediction(test_summary[i], phi_y, phi_k_y, vocab)
print(f"\tTest predictions computed in {time.process_time() - start} seconds")
    
train_prediction = numpy.empty(train_labels.shape, train_labels.dtype)
print("Computing Train Predictions...")
start = time.process_time()
for i in range(train_prediction.size):
    train_prediction[i] = prediction(train_summary[i], phi_y, phi_k_y, vocab)
print(f"\tTrain predictions computed in {time.process_time() - start} seconds")
    
# print(test_prediction) 
# print(train_prediction)

train_accuracy = 100* numpy.count_nonzero(train_labels==train_prediction)/train_labels.size
print("Train Set Accuracy by Model:", train_accuracy,"%")
test_accuracy = 100* numpy.count_nonzero(test_labels==test_prediction)/test_labels.size
print("Test Accuracy by Model:", test_accuracy,"%")


# store predictions
prediction_data = numpy.asarray(test_prediction)
numpy.savetxt('1_g_test_prediction.csv', prediction_data, delimiter=',')






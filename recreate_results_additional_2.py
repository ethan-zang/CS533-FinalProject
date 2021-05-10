import pickle
import nltk
from nltk.util import ngrams
import heapq
import numpy as np
import string
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2
import re
import sys
from typing import List, Dict


def get_token_frequency(tweet_set, flag):
    '''
    Utility method to calculate token frequencies for different representations.
    :param tweet_set: dataset of tweets
    :param flag: indicator to determine whether or not tokenization of the dataset is required
    :return: dictionary with all tokens and respective frequencies
    '''
    freqs = {}
    for tweet in tweet_set:
        if flag == "tokenize":
            tweet = nltk.word_tokenize(tweet)
        for word in tweet:
            if word in freqs.keys():
                freqs[word] += 1
            else:
                freqs[word] = 1
    return freqs


def create_frequency_vector(tweet_set, most_common_tokens, flag):
    '''
    Utility method to create vectors for most common tokens and their respective frequencies.
    :param tweet_set: dataset of tweets
    :param most_common_tokens: Heap queue of most common tokens to get frequencies of
    :param flag: indicator to determine whether or not tokenization of the dataset is required
    :return: Vector containing the most common tokens
    '''
    X = []
    for tweet_original in tweet_set:
        freq_vector = []
        for word in most_common_tokens:
            if flag == "tokenize":
                tweet = nltk.word_tokenize(tweet_original)
            else:
                tweet = tweet_original
            if word not in tweet:
                freq_vector.append(0)
            else:
                freq_vector.append(1)
        X.append(freq_vector)
    return np.asarray(X)


def unpickle(filepath):
    '''
    Utility function for unpickling datasets.
    :param filepath: path to .pkl file
    :return: loaded .pkl file
    '''
    with open(filepath, 'rb') as fp:
        unpickled = pickle.load(fp)
    return unpickled


def camel_case_split(str):
    '''
    Function to split apart camel case words
    :param str: input string
    :return: boolean value whether or not desired expression was found in input
    '''
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', str)


def switch_signal_feature():
    '''
    Function used to calculate and create extra features.
    :return: vector containing derived features for use in sentiment classification
    '''
    processed_lines = unpickle('../Dataset/dataset_humour_processed.pkl')
    signal_feature = []

    # Iterate through each key/tweet/label combo
    for key in processed_lines.keys():

        tweet = processed_lines[key]['tweet']

        # Instantiate features
        en_hi_switches = 0
        hi_en_switches = 0

        fraction_en = 0
        fraction_hi = 0

        total_words = 0
        for j in range(len(tweet)):

            # Get count of english and hindi words within a tweet
            if tweet[j].split("_")[1] == "en":
                fraction_en += 1
            elif tweet[j].split("_")[1] == "hi":
                fraction_hi += 1
            total_words += 1


            # Get number of switches between english and hindi and vice versa
            if j > 0 and tweet[j - 1].split("_")[1] == "en" and tweet[j].split("_")[1] == "hi":
                en_hi_switches += 1

            if j > 0 and tweet[j - 1].split("_")[1] == "hi" and tweet[j].split("_")[1] == "en":
                hi_en_switches += 1

        # Find total number of switches
        v = hi_en_switches + en_hi_switches

        # Calculate fraction
        fraction_en = fraction_en / (total_words * 1.00)
        fraction_hi = fraction_hi / (total_words * 1.00)

        pre_hi = [0] * (len(tweet) + 1)
        pre_en = [0] * (len(tweet) + 1)

        en_hi_vector = [0] * (len(tweet) + 1)
        hi_en_vector = [0] * (len(tweet) + 1)

        # Calculate en_hi_vector and hi_en_vector
        for i in range(len(tweet)):
            pre_hi[i + 1] = (tweet[i].split("_")[1] == 'hi') + pre_hi[i]
            pre_en[i + 1] = (tweet[i].split("_")[1] == 'en') + pre_en[i]

        for i in range(len(tweet)):
            if tweet[i].split("_")[1] == 'hi':
                en_hi_vector[i] = pre_en[i + 1]
            if tweet[i].split("_")[1] == 'en':
                hi_en_vector[i] = pre_hi[i + 1]

        # Calculate mean and std dev of en_hi_vector and hi_en_vector
        mean_en_hi = np.mean(en_hi_vector)
        stddev_en_hi = np.std(en_hi_vector)
        mean_hi_en = np.mean(hi_en_vector)
        stddev_hi_en = np.std(hi_en_vector)

        # New feature calculation
        has_switch = 0 if hi_en_switches == 0 and en_hi_switches == 0 else 1
        starts_ends = 1 if hi_en_switches == en_hi_switches else 0
        mean_hi_en_summed = np.mean(np.array(en_hi_vector) + np.array(hi_en_vector))
        mean_hi_en_switches = np.mean(hi_en_switches + en_hi_switches)

        # Switching mode combining
        signal_feature.append([en_hi_switches, hi_en_switches, v, fraction_en, fraction_hi, mean_hi_en, stddev_hi_en, mean_en_hi, stddev_en_hi])
    return signal_feature


def featureselection_add(features, train_tweets, train_truth):
    '''
    Build X_train and X_test input vectors in K-fold cross validation
    :param features: features selected
    :param train_tweets: training data set
    :param train_truth: Gold (ground truth) labels for training data
    :return: Recreated input X sets
    '''
    switching_feature_length = 9
    model = SelectKBest(score_func=chi2, k=num_features - switching_feature_length)
    train_tweets = np.array(train_tweets)
    features = np.array(features)
    train_tweets_small = train_tweets[:, :-switching_feature_length]
    switch_signal_feature = features[:, -switching_feature_length:]
    fit = model.fit(train_tweets_small, np.array(train_truth))
    train_features_reduced = fit.transform(features[:, :-switching_feature_length])
    train_final = np.hstack((train_features_reduced, switch_signal_feature))
    return train_final.tolist()


def train(X_train, X_test, y_train, y_test):
    '''
    Training and testing method for Support Vector Classifier
    :param X_train: input training data
    :param X_test: input testing data
    :param y_train: labels for training data
    :param y_test: labels for testing data
    '''
    # Create a Support Vector Classifier using linear kernel
    svc_classifier = SVC(kernel='linear', random_state=random_seed)
    # Fit SVC to training data
    svc_classifier.fit(X_train, y_train)
    print('train: ', svc_classifier.score(X_train, y_train), 'test: ', svc_classifier.score(X_test, y_test))
    y_pred = svc_classifier.predict(X_test)
    # Initialize metrics for F1 Score calculation
    false_negative = 0
    true_negative = 0
    false_positive = 0
    true_positive = 0
    # Iterate through label sets and count performance metrics
    for guess, truth in zip(y_pred, y_test):
        if guess == 0 and truth == 0:
            true_negative = true_negative + 1
        elif guess == 1 and truth == 1:
            true_positive = true_positive + 1
        elif guess == 0 and truth == 1:
            false_negative = false_negative + 1
        elif guess == 1 and truth == 0:
            false_positive = false_positive + 1
    # Calculate precision, recall, F1 score, and accuracy
    precision = float(true_positive) / (true_positive + false_positive)
    recall = float(true_positive) / (true_positive + false_negative)
    F1 = float(2 * precision * recall) / (precision + recall)
    accuracy = float(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    print("Acc", accuracy * 100, "F1", F1 * 100)
    # Append results from the current fold to the array for later calculation of average
    result.append([accuracy*100, F1*100, f1_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='micro')])


# Process system inputs
num_features = int(sys.argv[2])
random_seed = int(sys.argv[3])

# Unpickle dataset
lines = []
processed_lines = unpickle('../Dataset/dataset_humour_processed.pkl')
# print(processed_lines)

# Each line is a nested dictionary with 'tweet' and 'label' keys.
# Tweets are the words separated into a list
# Label is either 0/1 denoting humor classification
# Example: 687908334880423936: {'tweet': ['sober_en', 'ka_hi', 'phal_hi', 'weekend_en', 'hota_hi', 'hai_hi', '_ot'], 'label': 0}

for line in processed_lines.keys():
    lines.append(processed_lines[line])
    #print(processed_lines[line])

# Condense the tweets into a singular list
tweets = []
for i in lines:
    sentence = []
    for j in i['tweet']:
        sentence.append(j[:-3])
    tweets.append((' ').join(sentence))
# print(tweets[0])

#print(tweets)

# Tokenize and get count of all words in all tweets
word_frequency = get_token_frequency(tweets, "tokenize")
print(len(word_frequency))
# print(word2count)

# Extract {150} most common words
most_common_words = heapq.nlargest(num_features, word_frequency, key=word_frequency.get)
# print(most_common_words)

# BOW representation using 150 most common words
X_bow = create_frequency_vector(tweets, most_common_words, "tokenize")
#print(X_bow)

# Repeat previous process, but creating character level trigrams instead of BOW
# 'ne ', 'e v', ' vi', 'vio', 'iol', 'ole', 'len', 'enc', 'nce', 'ce ', 'e k' for example
n = 3
tweets_trigram = []
for tweet in tweets:
    trigram_sentence = []
    for i in range(len(tweet) - n + 1):
        trigram_sentence.append(tweet[i:i + n])
    tweets_trigram.append(trigram_sentence)

# print(tweets_trigram[0])

# Use helper method to get frequency of trigrams
word_frequency = get_token_frequency(tweets_trigram, "Tri")
print(len(word_frequency))
# Isolate most common trigrams and create frequency vector of the most common trigrams
most_common_trigrams = heapq.nlargest(num_features, word_frequency, key=word_frequency.get)
X_tri = create_frequency_vector(tweets_trigram, most_common_trigrams, "Tri")


# Unpickle file with tweetids
original = unpickle('../Dataset/data_with_tweetids.pkl')

# Seems to classify each word as Hi, En, or Ot
# print(original) #('stamps', 'En'), ('feet', 'En'), ('**', 'Ot'), ('acts', 'En'), ('like', 'En'), ('a', 'En'), ('dumb', 'En')

# Removes punctuation tokens
punctuation = string.punctuation
tweet_id = [int(i[0]) for i in original]
original = [[j[0] for j in i[1:-1] if j[0] not in punctuation] for i in original]
# print(original) #['Ek', 'galti', 'toh', 'sabko', 'maaf', 'hai', 'sarkaarpic', 'twitter', 'com/IFXsHrKdiY']

# Remove punctuation from words themselves
original_punctuation_stripped = []
for tweet in original:
    temp = []
    for word in tweet:
        flag = 0
        for letter in word:
            # Check if there is exists punctuation within the word
            if letter in punctuation:
                flag = 1
                break
        # Leading hashtag is special
        if word[0] == '#':
            flag = 2

        # Add word to temp if there is no punctuation
        if flag == 0:
            temp.append(word)
        elif flag == 1:
            for l in word.split(letter):
                temp.append(l)
        elif flag == 2:
            if len(camel_case_split(word)) == 0:
                temp.append(word[1:])
            else:
                for l in camel_case_split(word):
                    temp.append(l)
    temp = [tweet.lower() for tweet in temp if tweet != '']
    original_punctuation_stripped.append(temp)


original_dict = {}
for i, j in zip(tweet_id, original_punctuation_stripped):
    original_dict[i] = j

# Reconstruct original dictionary without punctuation
original = []
for i in processed_lines.keys():
    original.append(original_dict[i])

# Create X_hashtag vectors, same idea as BOW above
dataset_hash = [(' ').join(i) for i in original]
word_frequency = get_token_frequency(dataset_hash, "tokenize")
print(len(word_frequency))

freq_words = heapq.nlargest(num_features, word_frequency, key=word_frequency.get)
X_hashtag = create_frequency_vector(dataset_hash, freq_words, "tokenize")

X_signal_switch = switch_signal_feature()
X_signal_switch = np.array(X_signal_switch)

Y = np.array([i['label'] for i in lines])
X = np.hstack((X_bow, X_tri, X_hashtag, X_signal_switch))

result = []

# Divide training data into 10 folds
kf = KFold(n_splits=10)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_train_new = featureselection_add(X_train, X_train, y_train)
    X_test_new = featureselection_add(X_test, X_train, y_train)
    # print(len(X_train_new), len(X_train_new[0]), len(X_test_new), len(X_test_new[0]))
    train(X_train_new, X_test_new, y_train, y_test)

print('Acc: ', sum([i[0] for i in result]) / 10, 'F1: ', sum([i[1] for i in result]) / 10)


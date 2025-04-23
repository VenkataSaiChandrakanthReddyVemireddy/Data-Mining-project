# Mining text data.

import pandas as pd
from nltk.probability import FreqDist
import requests
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'coronavirus_tweets.csv'.
def read_csv_3(data_file):
	data = pd.read_csv(data_file,encoding = 'latin-1')
	return data

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	sentiments = df['Sentiment'].unique().tolist()
	return sentiments

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	count_sentiment = df['Sentiment'].value_counts()
	second_popular_sentiment = count_sentiment.index[1]
	return second_popular_sentiment

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	extremely_positive_tweets = df[df['Sentiment'] == 'Extremely Positive']
	count = extremely_positive_tweets.groupby('TweetAt')['Sentiment'].count()
	date_most_popular = count.idxmax()
	return date_most_popular

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet']=df['OriginalTweet'].str.replace(r'[^a-zA-Z\s]',' ',regex=True)
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split().str.join(' ')
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()
	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	count_with = tdf['OriginalTweet'].map(len).sum()
	return count_with

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	count_without = len(set(tdf['OriginalTweet'].explode()))
	return count_without

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	words_list = []
	for tkn in tdf['OriginalTweet']:
		for w in tkn:
			words_list.append(w)  
	
	frequency_dist = FreqDist(words_list)
	Sort_frequency_dist = sorted(frequency_dist.items(),key=lambda x:x[1],reverse=True)
	
	most_frequent = []
	for w,cnt in Sort_frequency_dist[:k]:
		most_frequent.append(w)
		
	return most_frequent 

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	req_stop_words = requests.get('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt')
	stp_words = req_stop_words.text.split('\n')
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [i for i in x if i not in stp_words and len(i)>2])
	return tdf

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	stemme = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [stemme.stem(i) for i in x])
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	w = df['OriginalTweet'].to_numpy()
	vect = CountVectorizer(ngram_range=(3,5))
	
	a = vect.fit_transform(w)
	val = df['Sentiment'].values
	
	clf = MultinomialNB()
	clf.fit(a,val)
	
	return clf.predict(a)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	acc = accuracy_score(y_true,y_pred)
	return round(acc,3)


print("Outputs: ")

data = read_csv_3(r'..\..\data\data\coronavirus_tweets.csv')
print(data)
sent = get_sentiments(data)
# print("Sentiments: " + str(sent))
sm_popular = second_most_popular_sentiment(data)
print("Second most popular: "+ str(sm_popular))
dm_popular = date_most_popular_tweets(data)
print("date most popular: "+ str(dm_popular))
d = lower_case(data)
print("lower case df: "+ str(d))
rem = remove_non_alphabetic_chars(d)
print("remove non alpha: "+ str(rem))
rmcw = remove_multiple_consecutive_whitespaces(rem)
print("remove multiple cons: "+ str(rmcw))
tok = tokenize(rmcw)
print("tokenize: "+ str(tok))
count_words = count_words_with_repetitions(tok)
print("count words: "+ str(count_words))
count_without_words = count_words_without_repetitions(tok)
print("count words without: "+ str(count_without_words))
freq = frequent_words(tok,5)
print("Freq words: "+ str(freq))
remove = remove_stop_words(tok)
print("Remove stop words: "+ str(remove))
ste = stemming(tok)
print("stem:"+ str(ste))
data = read_csv_3(r'..\..\data\data\coronavirus_tweets.csv')
m = mnb_predict(data)
print("predict: "+ str(m))
accu = mnb_accuracy(mnb_predict(data), data['Sentiment'])
print("Accuracy: "+ str(accu))






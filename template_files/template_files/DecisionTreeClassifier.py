# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
	data = pd.read_csv(data_file)
	data = data.drop('fnlwgt', axis=1)
	return data

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	n_rows = df.shape[0]
	return n_rows

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	l_columns = list(df.columns)
	return l_columns

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	m_values = df.isnull().values.sum()
	return m_values

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	return list(df.columns[df.isna().any()])

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
	df = (len(df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')])/len(df))*100
	return round(df,1)

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df = df.dropna()
	return df

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
	encode = pd.get_dummies(df.loc[:,df.columns != 'class'])
	return encode
# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	labelEncoding = preprocessing.LabelEncoder()
	s_encoded_values = labelEncoding.fit_transform(df['class'])
	series = pd.Series(s_encoded_values)
	return series

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	classifier = DecisionTreeClassifier()
	classifier.fit(X_train, y_train)
	panda_series = pd.Series(classifier.predict(X_train))
	return panda_series

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	error_rate = 1 - accuracy_score(y_true, y_pred)
	return error_rate


print("Outputs: ")
data = read_csv_1(r'..\..\data\data\adult.csv')
print(data)
n_rows = num_rows(data)
print("number of rows : " + str(n_rows))
c_names = column_names(data)
print("List of columns : " + str(c_names))
y = missing_values(data)
print("Missing values : " + str(y))
mis = columns_with_missing_values(data)
print("Columns with Missing values : " + str(mis))
bachelor = bachelors_masters_percentage(data)
print("Bachelors Masters percentage : " + str(bachelor))
updated_data = data_frame_without_missing_values(data)
print("Df without missing values : " + str(updated_data))
encode = one_hot_encoding(updated_data)
print("One hot encoding : " + str(encode))
le = label_encoding(updated_data)
print("label encoding : " + str(le))
x_train = encode
y_train = le
predict = dt_predict(x_train,y_train)
print("Predict : " + str(predict))
errorrate = dt_error_rate(predict,y_train)
print("Error rate : " + str(errorrate))

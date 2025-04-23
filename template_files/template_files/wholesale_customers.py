# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from IPython.display import display
import copy 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
    data = pd.read_csv(data_file)
    data = data.drop(['Channel','Region'],axis=1)
    return data

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	stat_dataframe = pd.DataFrame(columns=["mean","std","min","max"])
	for i in df.columns:
		mean = round(df[i].mean())
		standard_deviation = round(df[i].std())
		minimum = df[i].min()
		maximum = df[i].max()
		stat_dataframe.loc[i] = [mean,standard_deviation,minimum,maximum]
	
	return stat_dataframe

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	standardized_df = pd.DataFrame()
	for i in df.columns:
		att_value = (df[i]-df[i].mean())/df[i].std()
		standardized_df[i] = att_value
	return standardized_df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
def kmeans(df, k):
	k_mean = KMeans(init='random',n_clusters=k,n_init=10)
	k_mean.fit(df)
	return pd.Series(k_mean.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	k_mean_p = KMeans(init='k-means++',n_clusters=k,n_init=10)
	k_mean_p.fit(df)
	return pd.Series(k_mean_p.labels_)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	agglomerative_clust = AgglomerativeClustering(n_clusters=k)
	agglomerative_clust.fit(df)
	return pd.Series(agglomerative_clust.labels_)

# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.
def clustering_score(X,y):
	sol_score = sklearn.metrics.silhouette_score(X,y,metric = 'euclidean')
	return sol_score
# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	standardized_df = copy.deepcopy(df)
	standardized_df = standardize(df)
	Column_names = ['Algorithm','data','k','Silhouette Score']
	algorithms = ['Kmeans']*60 + ['Agglomerative']*6
	#print(algorithms)
	data_ent = (['Original'] + ['Standardized'])*33
	#print (data_ent)
	k_values = ([3]*20 + [5]*20 + [10]*20)+([3]*2 + [5]*2 + [10]*2)	
	#print (k_values)
	scores_kmeans = []
	scores_agglo = []
	values = [3,5,10]
	
	for i in values:
		c = 0
		for c in range(10):
			k_clust = clustering_score(df,kmeans(df,i))
			scores_kmeans.append(k_clust)
			k_clust_stan = clustering_score(standardized_df,kmeans(standardized_df,i))
			scores_kmeans.append(k_clust_stan)

	for j in values:
		agg_clust = clustering_score(df,agglomerative(df,j))
		scores_agglo.append(agg_clust)
		agg_clust_stan = clustering_score(standardized_df,agglomerative(standardized_df,j))
		scores_agglo.append(agg_clust_stan)

	silhou_scores = scores_kmeans + scores_agglo
	# print(len(silhou_scores))
	# print (silhou_scores)
	new_df = pd.DataFrame({'Algorithm':algorithms, 'data':data_ent, 'k':k_values, 'Silhouette Score':silhou_scores})
	return new_df  
# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	best_clust_score = rdf['Silhouette Score'].max()
	return best_clust_score

# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	k_means = KMeans(init='random', n_clusters = 3, n_init = 10)
	k_means.fit(df)
	column_names = df.columns
	
	with PdfPages('plots.pdf') as pdf:
		for i in range(len(column_names)):   
			for j in range(len(column_names)):
				if j>i:
					plt.scatter(df[column_names[i]],df[column_names[j]],c=k_means.labels_)
					plt.xlabel(column_names[i])
					plt.ylabel(column_names[j])
					pdf.savefig()


print("Outputs: ")
data = read_csv_2(r'..\..\data\data\wholesale_customers.csv')
#print(data)
summ = summary_statistics(data)
# print("Summary stats: " + str(summ))
stan = standardize(data)
#print("Standardized data: " + str(stan))
k =kmeans(data,5)
#print("K-Means: " + str(k))
agg = agglomerative(data,5)
#print("agglo: " + str(agg))
sil = clustering_score(data,k)
print("Sil score: " + str(sil))
a = cluster_evaluation(data)
print("Evaluation : " + str(a))
b = best_clustering_score(a)
print("Best score : " + str(b))
plot = scatter_plots(data)


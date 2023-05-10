# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:53:42 2023

@author: rcz
"""

# import modules
import sklearn.datasets as skdat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet

# import cluster tool
import cluster_tools as ct


def plot_cluster(gdp):
    
    # create new dataframe with only GDP per capita of all countries
    gdp = gdp.loc[gdp['Series Name'] == 'GDP per capita (current US$)']
    
    # set country name column as index
    gdp = gdp.set_index('Country Name')
    
    # slice dataframe
    gdp = gdp[['1990','1995','2000','2005','2010','2015']]
    
    # drop rows with ".." values
    gdp.drop(gdp.loc[gdp['1990']== ".."].index, inplace=True)
    gdp.drop(gdp.loc[gdp['1995']== ".."].index, inplace=True)
    gdp.drop(gdp.loc[gdp['2015']== ".."].index, inplace=True)
    
    # convert values into float
    gdp = gdp.astype(float)
    
    # find correlation matrix
    corr = gdp.corr()
    print(corr)
    
    # plot
    pd.plotting.scatter_matrix(gdp, figsize=(12, 12), s=5, alpha=0.8)
    
    # extract the two columns for clustering
    gdp_ex = gdp[["1990", "2015"]]
    
    # normalise, store minimum and maximum
    gdp_norm, gdp_min, gdp_max = ct.scaler(gdp_ex)
    
    print("cluster number  | silhouette score")
    
    # loop over number of clusters
    for ncluster in range(2, 10):
        
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)
    
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(gdp_norm)     # fit done on x,y pairs
    
        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
    
        # calculate the silhoutte score
        print(ncluster, "\t\t\t\t|",skmet.silhouette_score(gdp_ex, labels))
        
    # best number of clusters
    ncluster = 8 
    
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(gdp_norm)     # fit done on x,y pairs
    
    labels = kmeans.labels_
        
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    
    # convert cen to a numpy array
    cen = np.array(cen)
    
    # extract x coordinates of cluster centres
    xcen = cen[:, 0]
    
    # extract y coordinates of cluster centres
    ycen = cen[:, 1]
    
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    
    # set colormap
    cm = plt.cm.get_cmap('tab10')
    
    # plot values
    plt.scatter(gdp_norm["1990"], gdp_norm["2015"], 10, labels, marker="o", cmap=cm)
    
    # plot cluster centres
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    
    #set labels
    plt.xlabel("GDP per head(1990)")
    plt.ylabel("GDP per head(2015)")
    
    plt.show()
    
    return # function must finish with return


# read data into dataframe
gdp = pd.read_csv('data.csv')

# call plot_cluster function
plot_cluster(gdp)
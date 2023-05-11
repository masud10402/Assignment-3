# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:53:42 2023

@author: Masud Rana
"""

# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

# from sklearn import cluster
import sklearn.cluster as cluster
import sklearn.metrics as skmet

import cluster_tools as ct
import errors as err

import warnings
warnings.filterwarnings('ignore')


# reading function
def get_file():
    """
    This function will read the dataframe and return it. It accepts no argument.
    """

    df = pd.read_csv("data.csv")

    return df


# define function for producing cluster
def plot_cluster(df):
    """
    This function will produce the cluster plot. Tt will accept one datafame as argument.
    """

    # create new dataframe with only co2 per capita of all countries
    co2 = df.loc[df['Series Name'] == 'CO2 emissions (kt)']

    # set country name column as index
    co2 = co2.set_index('Country Name')

    # slice dataframe
    co2 = co2[['1990', '1998', '2005', '2013', '2019']]

    # drop rows with ".." values
    co2.drop(co2.loc[co2['1990'] == ".."].index, inplace=True)
    co2.drop(co2.loc[co2['1998'] == ".."].index, inplace=True)

    # convert values into float
    co2 = co2.astype(float)

    # extract the two columns for clustering
    co2_ex = co2[["1990", "2019"]]

    # normalise, store minimum and maximum
    co2_norm, co2_min, co2_max = ct.scaler(co2_ex)

    print("cluster number  | silhouette score")

    # loop over number of clusters
    for ncluster in range(2, 10):

        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the data, results are stored in the kmeans object
        kmeans.fit(co2_norm)     # fit done on x,y pairs

        labels = kmeans.labels_

        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_

        # calculate the silhoutte score
        print(ncluster, "\t\t\t\t|", skmet.silhouette_score(co2_ex, labels))

    # best number of clusters
    ncluster = 7

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, random_state=0)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(co2_norm)     # fit done on x,y pairs

    labels = kmeans.labels_

    # create a new column with cluster labels
    co2_ex['cluster'] = labels

    # create different dataframe for each cluster
    cluster0 = co2_ex.loc[co2_ex['cluster'] == 0]
    cluster1 = co2_ex.loc[co2_ex['cluster'] == 1]
    cluster2 = co2_ex.loc[co2_ex['cluster'] == 2]
    cluster3 = co2_ex.loc[co2_ex['cluster'] == 3]
    cluster4 = co2_ex.loc[co2_ex['cluster'] == 4]
    cluster5 = co2_ex.loc[co2_ex['cluster'] == 5]
    cluster6 = co2_ex.loc[co2_ex['cluster'] == 6]
    cluster7 = co2_ex.loc[co2_ex['cluster'] == 7]

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    # convert cen to a numpy array
    cen = np.array(cen)

    # Applying the backscale function to convert the cluster centre
    scen = ct.backscale(cen, co2_min, co2_max)

    # extract real values of x coordinates of cluster centres
    xcen = scen[:, 0]

    # extract real values of y coordinates of cluster centres
    ycen = scen[:, 1]

    # produce a plot of points and fit
    plt.figure(figsize=(8.0, 8.0))

    # set colormap
    cm = plt.cm.get_cmap('tab10')

    # plot points
    plt.scatter(co2_ex["1990"], co2_ex["2019"],
                10, labels, marker="o", cmap=cm)

    # plot cluster centres
    plt.scatter(xcen, ycen, 45, "k", marker="d")

    # set labels
    plt.xlabel("co2 emission (kt) (1990)")
    plt.ylabel("co2 emission (kt) (2019)")

    # set title
    plt.title("Clustering of all countries co2 emission")

    # save png
    plt.savefig('cluster.png', dpi=300, bobox_inches='tight')

    plt.show()

    return  # function must finish with return


# define a function to create fit models for gdp
def fit_model_gdp(df, country):
    """
    This function will produce simple models fitting data using curve_fit.
    
    It will creat model for GDP indicator.

    It will receive two arguments:
    df: one dataframe
    country: country name as a string
    """

    # create new dataframe with the rows where country name is China
    china = df.loc[df['Country Name'] == country]

    # create dataframe only with Value
    china = china.loc[china['Series Name'] == "GDP (current US$)"]

    # drop unwanted columns
    china = china.drop(columns=['Country Name', 'Series Name'])

    # Transpose the dataframe
    china = china.T.copy()

    # reset index
    china = china.reset_index()

    # rename columns
    china.columns = ['Year', "GDP"]

    # drop rows with values ".."
    china.drop(china.loc[china["GDP"] == ".."].index, inplace=True)

    # convert values from string to integar
    china['Year'] = pd.to_numeric(china['Year'])
    china["GDP"] = pd.to_numeric(china["GDP"])

    # define a function to calculate exponential function

    def exponential(t, n0, g):
        """Calculates exponential function with scale factor n0 and growth rate g."""

        t = t - 1990
        f = n0 * np.exp(g*t)

        return f

    # do the fitting
    # curvefit expects the argument list of the fit function to be (x, p1, p2, p3, p4, ....)
    param, covar = opt.curve_fit(
        exponential, china["Year"], china["GDP"], p0=(1.7e10, 0.03))

    # lets get the std. dev.
    # np.diag() extracts the diagonal of a matrix
    sigma = np.sqrt(np.diag(covar))

    # set fit as a new column
    china["fit"] = exponential(china["Year"], *param)

    # set year with future ten years
    year = np.arange(1960, 2031)

    # set exponential model with future ten years
    forecast = exponential(year, *param)

    # calculate lower and upper limits of confidence interval
    low, up = err.err_ranges(year, exponential, param, sigma)

    # produce a plot of points and fit
    plt.figure()

    # plot Value
    plt.plot(china["Year"], china["GDP"], label="GDP")

    # plot model
    plt.plot(year, forecast, label="forecast")

    # plot confidence range
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)

    # set labels
    plt.xlabel("Year")
    plt.ylabel("GDP")

    # set title
    plt.title(f"GDP of {country}")

    # set legend
    plt.legend(loc='upper center', ncol=2, frameon=False)

    # save png
    plt.savefig('austria_gdp.png', dpi=300, bobox_inches='tight')

    plt.show()

    return  # function finishes with return


# define a function to create fit models for gdp
def fit_model_forest(df, country):
    """
    This function will produce simple models fitting data using curve_fit.
    
    It will creat model for Forest area indicator.

    It will receive two arguments:
    df: one dataframe
    country: country name as a string
    """

    # create new dataframe with the rows where country name is China
    china = df.loc[df['Country Name'] == country]

    # create dataframe only with Value
    china = china.loc[china['Series Name'] == "Forest area (% of land area)"]

    # drop unwanted columns
    china = china.drop(columns=['Country Name', 'Series Name'])

    # Transpose the dataframe
    china = china.T.copy()

    # reset index
    china = china.reset_index()

    # rename columns
    china.columns = ['Year', "Forest"]

    # drop columns with values ".."
    china.drop(china.loc[china["Forest"] == ".."].index, inplace=True)

    # convert values from string to integar
    china['Year'] = pd.to_numeric(china['Year'])
    china["Forest"] = pd.to_numeric(china["Forest"])

    # define a function to calculate exponential function

    def exponential(t, n0, g):
        """Calculates exponential function with scale factor n0 and growth rate g."""

        t = t - 1990
        f = n0 * np.exp(g*t)

        return f

    # do the fitting
    # curvefit expects the argument list of the fit function to be (x, p1, p2, p3, p4, ....)
    param, covar = opt.curve_fit(
        exponential, china["Year"], china["Forest"], p0=(1.7e10, 0.03))

    # lets get the std. dev.
    # np.diag() extracts the diagonal of a matrix
    sigma = np.sqrt(np.diag(covar))

    # set fit as a new column
    china["fit"] = exponential(china["Year"], *param)

    # set year with future ten years
    year = np.arange(1990, 2031)

    # set exponential model with future ten years
    forecast = exponential(year, *param)

    # calculate lower and upper limits of confidence interval
    low, up = err.err_ranges(year, exponential, param, sigma)

    # produce a plot of points and fit
    plt.figure()

    # plot Value
    plt.plot(china["Year"], china["Forest"], label="forest area")

    # plot model
    plt.plot(year, forecast, label="forecast")

    # plot confidence range
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)

    # set labels
    plt.xlabel("Year")
    plt.ylabel("Forest area (% of land area)")

    # set title
    plt.title(f"Forest area of {country}")

    # set legend
    plt.legend(loc='upper center', ncol=2, frameon=False)

    # save png
    plt.savefig('austria_forest.png', dpi=300, bobox_inches='tight')

    plt.show()

    return  # function finishes with return


# call the get_file function
df = get_file()

# call plot_cluster function
plot_cluster(df)

# I have chose three countries cluster. Bangladesh and Austria from cluster0 and China from cluster5.
# We will create model fitting data using curve_fit for these three countries.
# call fit model gdp fucntion
fit_model_gdp(df, 'China')
fit_model_gdp(df, 'Bangladesh')
fit_model_gdp(df, 'Austria')

# call fit model forest function
fit_model_forest(df, 'China')
fit_model_forest(df, 'Bangladesh')
fit_model_forest(df, 'Austria')

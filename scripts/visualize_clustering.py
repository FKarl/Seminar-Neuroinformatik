import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def kmeans_clustering(csv_file, number_clusters):
    """
    Clustering and visualizationusing KMeans

    :param csv_file: String, path to the csv file
    :param number_clusters: int, number of clusters
    """
    # read the data set and drop labels
    smiley_df = pd.read_csv(csv_file)
    X = smiley_df.drop('label', axis=1).values

    kmeans = KMeans(n_clusters=number_clusters)
    kmeans.fit(X)
    predicted_cluster = kmeans.predict(X)

    # plotting the clusters
    plt.scatter(X[:, 0], X[:, 1], c=predicted_cluster, cmap='rainbow')
    plt.savefig('smiley_kmeans.svg', format='svg')
    plt.show()


def dbscan_clustering(csv_file, eps=0.5, min_samples=5):
    """
    Clustering and visualization using DBSCAN

    :param csv_file: String, path to the csv file
    :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    :param min_samples: The Minimum number of samples in a neighborhood for a point to be considered as a core point.
    """
    # read the data set and drop labels
    df = pd.read_csv(csv_file)
    X = df.drop('label', axis=1).values

    # eps = maximum distance between two samples for them to be considered as in the same neighborhood
    # min_samples = minimum number of samples in a neighborhood for a point to be considered as a core point
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    labels = dbscan.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
    plt.savefig('smiley_dbscan.svg', format='svg')
    plt.show()


def plot_csv(csv_file):
    """
    Plot the data set without any clustering

    :param csv_file: String, path to the csv file
    """
    df = pd.read_csv(csv_file)
    X = df.drop('label', axis=1).values

    plt.scatter(X[:, 0], X[:, 1], color='black')
    plt.savefig('smiley.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    # read location of the csv file
    csv_filename = input('Enter the path of the csv file: ') or 'smiley.csv'

    # plot the csv
    plot_csv(csv_filename)
    kmeans_clustering(csv_filename, 4)
    dbscan_clustering(csv_filename)

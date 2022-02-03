import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def kmeans_clustering(csv_file, number_clusters):
    """
    Clustering and visualization using KMeans

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
    plt.savefig(csv_file.removesuffix('.csv') + '_kmeans.svg', format='svg')
    plt.show()


def kmeans_clustering_with_centroids(csv_file, number_clusters):
    """
    Clustering and visualization with centroids using KMeans
    :param csv_file: String, path to the csv file
    :param number_clusters: int, number of clusters
    """
    df = pd.read_csv(csv_file)
    X = df.drop('label', axis=1).values

    # KMeans but plot all learning steps
    kmeans = KMeans(n_clusters=number_clusters)
    kmeans.fit(X)
    predicted_cluster = kmeans.predict(X)

    # plot the cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=predicted_cluster, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', label='centroids', marker='x',
                s=150, linewidths=3)
    plt.legend()
    plt.savefig(csv_file.removesuffix('.csv') + '_kmeans_with_centroids.svg', format='svg')
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
    plt.savefig(csv_file.removesuffix('.csv') + '_dbscan.svg', format='svg')
    plt.show()


def plot_csv(csv_file):
    """
    Plot the data set without any clustering

    :param csv_file: String, path to the csv file
    """
    df = pd.read_csv(csv_file)
    X = df.drop('label', axis=1).values

    plt.scatter(X[:, 0], X[:, 1], color='black')
    plt.savefig(csv_file.removesuffix('csv') + 'svg', format='svg')
    plt.show()


if __name__ == '__main__':
    # read location of the csv file
    csv_filename = input('Enter the path of the csv file: ') or 'smiley.csv'
    number_of_clusters = int(input('Enter the number of clusters: ') or '4')

    # plot the csv
    plot_csv(csv_filename)
    kmeans_clustering(csv_filename, number_of_clusters)
    kmeans_clustering_with_centroids(csv_filename, number_of_clusters)
    dbscan_clustering(csv_filename)

import pandas as pd
import numpy as np


def create_random_cluster(number_of_clusters, csv_file='random.csv'):
    """
    Creates a specified number of random clusters and writes them to a csv file.

    :param number_of_clusters: number of clusters to create
    :param csv_file: String, name of the csv file to create
    """
    rows = []

    for i in range(number_of_clusters):
        # calc center
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 50)

        # calc radius of cluster
        r = np.random.uniform(5, 20)

        # calc number of points
        n = np.random.randint(30, 120)

        x_new = np.random.normal(x, scale=10, size=n)
        y_new = np.random.normal(y, scale=10, size=n)

        # add points to dataframe
        rows.append(pd.DataFrame({'x': x_new, 'y': y_new, 'label': i}))

    # shuffle and save
    df = pd.concat(rows)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(csv_file, index=False)


if __name__ == '__main__':
    number_of_clusters = int(input('Enter number of clusters: ') or 3)
    create_random_cluster(number_of_clusters)

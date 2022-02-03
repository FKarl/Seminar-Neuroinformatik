import pandas as pd
import numpy as np


def create_smiley(csv_filename='smiley.csv'):
    """
    Create a data set in the shape of a smiley face
    and save the data set as a csv file at the location specified by csv_filename.

    :param csv_filename: String, path in which the result should be saved
    """
    # A list of all rows in the data set
    rows = []

    # init values for the smiley
    eye_points = 150
    eye_coords_l = [3.0, 7.0]
    eye_coords_r = [6.0, 7.0]

    smile_points = 500
    smile_center = [4.5, 4.0]
    smile_width = 1.0
    smile_bending = 1.5

    head_points = 1000
    head_center = [4.5, 5.0]
    head_radius = 4.0

    noise_points = 50
    noise_range = [0.0, 10.0]

    random_scale = 0.2

    # draw eyes
    for i in range(eye_points):
        x_l = np.random.normal(eye_coords_l[0], scale=random_scale)
        y_l = np.random.normal(eye_coords_l[1], scale=random_scale)
        x_r = np.random.normal(eye_coords_r[0], scale=random_scale)
        y_r = np.random.normal(eye_coords_r[1], scale=random_scale)
        rows.append({'x': x_l, 'y': y_l, 'label': 'left_eye'})
        rows.append({'x': x_r, 'y': y_r, 'label': 'right_eye'})

    # draw smile
    for i in range(smile_points):
        r = np.random.normal(smile_width, scale=random_scale)
        # a bit more than half a circle
        part = np.random.uniform(- 0.6 * np.pi, 0.6 * np.pi)

        x = np.sin(part) * r * smile_bending + smile_center[0]
        y = - np.cos(part) * r + smile_center[1]
        rows.append({'x': x, 'y': y, 'label': 'smile'})

    # draw head
    for i in range(head_points):
        part = np.random.uniform(- np.pi, np.pi)
        r = np.random.normal(head_radius, scale=random_scale)
        x = np.sin(part) * r + head_center[0]
        y = - np.cos(part) * r + head_center[1]
        rows.append({'x': x, 'y': y, 'label': 'head'})

    # draw noise
    for i in range(noise_points):
        x = np.random.uniform(noise_range[0], noise_range[1])
        y = np.random.uniform(noise_range[0], noise_range[1])
        rows.append({'x': x, 'y': y, 'label': 'noise'})

    # shuffle and save
    df = pd.DataFrame(rows)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(csv_filename, index=False)


if __name__ == '__main__':
    create_smiley()

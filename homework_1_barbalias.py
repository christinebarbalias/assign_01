import time
from collections import defaultdict
from datetime import timedelta

import imageio
import matplotlib.image
import numpy as np


def adjust_centroids(pixel, cluster_assign):
    number_rows = pixel.shape[0]
    number_cols = pixel.shape[1]
    euclidian_distance = 2
    new_centroids = defaultdict()
    final_centroids = []

    for i in range(number_rows):
        for j in range(number_cols):
            pix = pixel[i, j]
            cluster = cluster_assign[i, j]
            if cluster not in new_centroids:
                new_centroids[cluster] = list()

            new_centroids[cluster].append(pix)

    for cluster in new_centroids:
        centroid_list = np.array(new_centroids[cluster])

        if euclidian_distance == 1:
            updated_centroid = np.median(centroid_list, axis=0).astype(
                np.uint8)
        else:
            updated_centroid = np.around(
                centroid_list.sum(axis=0) / len(centroid_list)).astype(
                np.uint8)

        final_centroids.append(updated_centroid.tolist())

    return final_centroids


def assign_cluster(pix, c):
    min_distance = np.inf
    cluster_index = 0
    euclidian_distance = 2

    for i in range(len(c)):
        x = pix - c[i]

        if euclidian_distance == 1:
            distance = np.linalg.norm(x, 1)
        else:
            distance = np.linalg.norm(x, 2)

        if i == 0:
            min_distance = distance
            cluster_index = i
        else:
            if distance < min_distance:
                cluster_index = i
                min_distance = distance
    return cluster_index


def k_means(pixel, k):
    start_time = time.monotonic()

    number_rows = pixel.shape[0]
    number_cols = pixel.shape[1]
    col_channels = pixel.shape[2]

    np.random.seed(1)
    color_red = np.random.randint(0, 255, size=k)
    color_green = np.random.randint(0, 255, size=k)
    color_blue = np.random.randint(0, 255, size=k)
    centroids = np.array(
        list(zip(color_red, color_green, color_blue))).tolist()

    cluster_assign = np.empty(shape=(number_rows, number_cols), dtype='uint8')

    loop_count = 1

    for i in range(number_rows):
        for j in range(number_cols):
            pix = pixel[i][j]
            cluster_assign[i, j] = assign_cluster(pix, centroids)

    new_centroids = sorted(adjust_centroids(pixel, cluster_assign))

    while True:
        loop_count += 1

        centroids = new_centroids

        for i in range(number_rows):
            for j in range(number_cols):
                pix = pixel[i][j]
                cluster_assign[i, j] = assign_cluster(pix, centroids)

        new_centroids = adjust_centroids(pixel, cluster_assign)
        if centroids == new_centroids:
            break

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time).seconds

    final_image = np.empty(shape=(number_rows, number_cols, col_channels),
                           dtype='object')
    for row in range(number_rows):
        for col in range(number_cols):
            pix = cluster_assign[row][col]
            final_image[row, col] = np.array(new_centroids[pix])

    cluster_labels = cluster_assign + 1
    final_image = np.reshape(final_image, (pixel.shape))

    return cluster_labels, new_centroids, final_image, loop_count, time_diff


if __name__ == "__main__":
    k_vals = [2, 4]
    images = ['data/Glockenbronze.png',
             'data/football.bmp']  # , 'data/bogart.jpg']
    for k in k_vals:
        for image in images:
            original_image = (imageio.v2.imread(image))
            cluster_label, cluster_center, comp_image, iterations, run_time = k_means(original_image, k)
            image_name = image.split('/')[1].split('.')[0]
            matplotlib.image.imsave(
                'Q2_output/' + image_name + '_' + str(k) + '.png',
                comp_image.astype(np.uint8))
            print(image_name + ', k = ' + str(k) + ', Iterations = ' + str(
                iterations) + ', Run Time = ' + str(run_time) + ' secs.')
import time
from collections import defaultdict
from datetime import timedelta

import imageio
import matplotlib.image
import numpy as np


def adjust_centroids(pixel, cluster_assign):
    euclidian_distance = 2
    number_rows, number_cols = pixel.shape[0], pixel.shape[1]
    new_centroids = defaultdict(list)

    for i in range(number_rows):
        for j in range(number_cols):
            current_pixel = pixel[i, j]
            current_cluster = cluster_assign[i, j]
            new_centroids[current_cluster].append(current_pixel)

    final_centroids = create_centroids(new_centroids, euclidian_distance)
    return final_centroids


def create_centroids(centroid_map, distance_measure):
    centroids = []
    for cluster in centroid_map:
        pixels_in_cluster = np.array(centroid_map[cluster])
        if distance_measure == 1:
            updated_centroid = np.median(pixels_in_cluster, axis=0).astype(
                np.uint8)
        else:
            updated_centroid = np.around(
                pixels_in_cluster.sum(axis=0) / len(pixels_in_cluster)).astype(
                np.uint8)

        centroids.append(updated_centroid.tolist())

    return centroids


def get_new_centroids(centroids, cluster_assign, number_cols, number_rows,
                      pixel):
    for i in range(number_rows):
        for j in range(number_cols):
            pix = pixel[i][j]
            cluster_assign[i, j] = assign_cluster(pix, centroids)
    new_centroids = sorted(adjust_centroids(pixel, cluster_assign))
    return new_centroids


def assign_cluster(pix, centroids):
    min_distance = np.inf
    cluster_index = 0
    for i, centroid in enumerate(centroids):
        distance = np.linalg.norm(pix - centroid)
        if distance < min_distance:
            cluster_index = i
            min_distance = distance
    return cluster_index


def k_means(image_pixels, cluster_count):
    LOWEST_RGB_VALUE = 0
    HIGHEST_RGB_VALUE = 255
    np.random.seed(1)

    def create_initial_centroids():
        color_red = np.random.randint(LOWEST_RGB_VALUE, HIGHEST_RGB_VALUE,
                                      size=cluster_count)
        color_green = np.random.randint(LOWEST_RGB_VALUE, HIGHEST_RGB_VALUE,
                                        size=cluster_count)
        color_blue = np.random.randint(LOWEST_RGB_VALUE, HIGHEST_RGB_VALUE,
                                       size=cluster_count)
        return np.array(list(zip(color_red, color_green, color_blue))).tolist()

    start_time = time.monotonic()
    image_shape = image_pixels.shape
    number_rows, number_cols, col_channels = image_shape
    centroids = create_initial_centroids()
    cluster_assign = np.empty(shape=(number_rows, number_cols), dtype='uint8')

    iteration_count = 1
    new_centroids = get_new_centroids(centroids, cluster_assign, number_cols,
                                      number_rows, image_pixels)
    while centroids != new_centroids:
        iteration_count += 1
        centroids = new_centroids
        new_centroids = get_new_centroids(centroids, cluster_assign,
                                          number_cols, number_rows,
                                          image_pixels)

    end_time = time.monotonic()
    processing_time_secs = timedelta(seconds=end_time - start_time).seconds

    final_image = np.empty(shape=image_shape, dtype='object')
    for i in range(number_rows):
        for j in range(number_cols):
            final_image[i, j] = np.array(new_centroids[cluster_assign[i, j]])
    final_image = np.reshape(final_image, image_shape)

    return final_image, iteration_count, processing_time_secs



def main():
    k_vals = [2, 4, 6]
    images = ['data/Glockenbronze.png',
              'data/football.bmp']  # , 'data/bogart.jpg']
    for each_k in k_vals:
        for image in images:
            original_image = (imageio.v2.imread(image))
            comp_image, iterations, run_time = k_means(original_image, each_k)
            image_name = image.split('/')[1].split('.')[0]
            matplotlib.image.imsave(
                'Q2_output/' + image_name + '_' + str(each_k) + '.png',
                comp_image.astype(np.uint8))
            print(f"{image_name=}, {each_k=}, {iterations=}, {run_time=} sec")


if __name__ == "__main__":
    main()
import scipy.io as sio
import numpy as np
from scipy.stats import mode
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist
import copy
import time
data = sio.loadmat("mnist_10digits.mat")
xtrain = data["xtrain"].T
ytrain = data["ytrain"].T
xtrain = xtrain/255
K = 10
np.random.seed(100)
def euclidean_assignment(centroids, data):
    c2 = np.sum(np.power(centroids, 2), axis=0, keepdims=True)
    tmpdiff = (2 * np.dot(data.T, centroids) - c2)
    labels = np.argmax(tmpdiff, axis=1).squeeze()
    return labels
def manhattan_assignment(centroids, data):
    ncentroids = centroids[:, :, np.newaxis]
    ndata = data[:, np.newaxis, :]
    diff = np.abs(ncentroids - ndata)
    sums = np.sum(diff, axis = 0)
    labels = np.argmin(sums, axis = 0).squeeze()
    return labels
def euclidean_centroids(labels, data, k):
    D, N = data.shape
    final_labels = csc_matrix((np.ones(N), (np.arange(0, N, 1), labels)), shape=(N,
k))
    count = final_labels.sum(axis=0)
    centroids = np.array((final_labels.T.dot(data.T)).T / count)
    zero_ind = np.argwhere(count == 0)
    if zero_ind.shape[0] >= 0:
        for i in zero_ind:
            centroids[:, i] = np.random.uniform(0, 1, size=D)
    return centroids
def manhattan_centroids(labels, data, k):
    dim, _ = data.shape
    centroids = np.zeros((dim, k))
    for i in range(k):
        centroids[:, i] = np.median(data[:, labels == i], axis=1) if np.sum(labels
                                                                            == i) > 0 else np.random.uniform(0, 1,
                                                                                                             size=D)
    return centroids


def k_means(input_data, k=K, euclidean=True):
    if euclidean:
        label_func = euclidean_assignment
        centroid_func = euclidean_centroids
    else:
        label_func = manhattan_assignment
        centroid_func = manhattan_centroids


centroids = np.random.uniform(0, 1, size=(input_data.shape[0], K))
i = 1
start = time.time()
diff = 9999999
while True:
    print(f"--iteration {i}, diff={diff}")
    i += 1
    old_centroids = copy.deepcopy(centroids)
    labels = label_func(centroids, input_data)
    centroids = centroid_func(labels, input_data, k)
    diff = np.linalg.norm(centroids - old_centroids, ord='fro')
    if diff <= 1e-6:
        break
    if i == 100:
        break
end = time.time()
print(f"Kmeans for {'euclidean' if euclidean else 'manhattan'} took {end -start} seconds")
return labels


def match_results(trained_clusters, true_data, num_clusters = K):
    total = 0
    for i in range(num_clusters):
        true_cluster_i = true_data[trained_clusters == i]
        if true_cluster_i.size == 0:
            continue
        most_common_member = mode(true_cluster_i).mode
        num = (true_cluster_i == most_common_member).sum()
        total += num
        result = num / true_cluster_i.shape[0]
        print(f"Cluster {i + 1} has a purity score of {result * 100} percent")
        print(f"The total score overall is {(total / 60000) * 100} percent")
    for is_euclidean in [True, False]:
        print(f"Testing {'Euclidean' if is_euclidean else 'Manhattan'}")
        for attempt in range(3):
            try:
                clusters = k_means(xtrain, euclidean=is_euclidean)
                break
            except Exception as e:
                print(f"Failure with reason: {str(e)} on attempt {attempt + 1}/3")
                if attempt == 2:
                    raise e
        match_results(clusters, ytrain.squeeze(), K)
import numpy as np
Laplacian_matrix = np.array([
    [ 2, -1, -1,  0,  0],
    [-1,  2, -1,  0,  0],
    [-1, -1,  2,  0,  0],
    [ 0,  0,  0,  1, -1],
    [ 0,  0,  0, -1,  1]
])
eigenval, eigenvectors = np.linalg.eig(Laplacian_matrix)
print("Eigenval:", eigenval)
print("Eigenvectors:\n",eigenvectors)

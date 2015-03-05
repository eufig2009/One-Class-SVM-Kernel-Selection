import numpy as np
import pylab as pl

from scipy.linalg import cholesky



def generate_synthetic_data(size, dim=2, centroid_count=1, centroid_dispersion=1):
    size = int(size)
    centroids = np.random.rand(centroid_count, dim) * 10 - 5
    all_data_part = []
    for point in centroids:
        covariation_matrix = np.random.randn(dim, dim)
        covariation_matrix = np.dot(covariation_matrix, covariation_matrix.T)
        covariation_matrix = cholesky(covariation_matrix)
        data_part = np.random.randn(int(size / centroid_count), dim)
        data_part = np.dot(data_part, covariation_matrix) + point
        all_data_part.append(data_part)
    data = np.concatenate(all_data_part, axis=0)
    return data


def generate_outlier(size, dim=2, space=1000):
    size = int(size)
    outliers = (np.random.rand(size, dim) - 0.5) * space * 2
    return outliers


if __name__ == '__main__':
    size = 1000
    outliers_part = 0.1
    dim = 2
    centroid_count = 5
    centroid_dispersion = 1
    inliers = generate_synthetic_data(size * (1 - outliers_part), 
                                               dim, centroid_count, 
                                               centroid_dispersion)
    pl.scatter(inliers[:, 0], inliers[:, 1], marker='o', label='normal')
    outliers = generate_outlier(size*outliers_part, dim, space=centroid_dispersion*50)
    pl.scatter(outliers[:, 0], outliers[:, 1], marker='s', c='none', label='anomaly')
    pl.legend(loc='best')
    pl.show()
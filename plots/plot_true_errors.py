import numpy as np
import pandas as pd
import pylab as pl

from sklearn.svm import SVDD
from scipy.linalg import cholesky
#from sklearn.metrics import roc_auc_score
from random import sample
from sklearn.cross_validation import KFold


def generate_outlier(size, dim=2, space=1000):
	'''
	Generate outliers from uniform distribution
	'''
    size = int(size)
    outliers = (np.random.rand(size, dim) - 0.5) * space * 2
    return outliers

def generate_synthetic_data(size, dim=2, centroid_count=2, centroid_dispersion=1):
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

def generate_dataset(size, outliers_part=0.12, dim=2, centroid_count=2, centroid_dispersion=1):
    inliers = generate_synthetic_data(size * (1 - outliers_part), 
                                               dim, centroid_count, 
                                               centroid_dispersion)
    outliers = generate_outlier(size*outliers_part, dim, space=centroid_dispersion*50)
    data = np.concatenate([inliers, outliers], axis=0)
    data = pd.DataFrame(data)
    data['label'] = ['target'] * len(inliers) + ['outlier'] * len(outliers)
    return data

all_gammas = np.logspace(-10, 10, 50)

def split_anomaly_normal_data(data, outliers_fraction=0.1):
    outliers = data.query("label == 'outlier'")
    inliers = data.query("label == 'target'")
    outliers_count = int(len(inliers) * outliers_fraction / (1 - outliers_fraction))
    if outliers_count > len(outliers):
        raise ValueError("There are no so many outliers")
    outliers_rows = sample(outliers.index, outliers_count)
    selected_outliers = outliers.ix[outliers_rows]
    return inliers.iloc[:, :-1], selected_outliers.iloc[:, :-1]

def split_data_set(data, parts_count=3):
    all_parts = []
    kfold = KFold(len(data), n_folds=parts_count)
    for train, test in kfold:
        all_parts.append(data.iloc[test, :])
    return all_parts

def calculate_errors(all_gammas):
    data = generate_dataset(1000, dim=2, centroid_count=2)
    normal_data, anomaly_data = split_anomaly_normal_data(data)
    normal_train, normal_validate, normal_test = split_data_set(normal_data, 3)
    anomaly_train, anomaly_validate, anomaly_test = split_data_set(anomaly_data, 3)
    values = []
    for gamma in all_gammas:
        C = 1.0 / (0.1 * (len(data)))

        clf = SVDD(kernel='rbf', C=C, gamma=gamma)
        clf.fit(np.concatenate([normal_validate, anomaly_validate], axis=0))
        #prediction_normal = clf.decision_function(normal_test)
        prediction_anomaly = clf.decision_function(anomaly_test)
        print prediction_anomaly
        #predictions = -np.concatenate([prediction_normal, prediction_anomaly])
        #true_labels = [-1] * len(prediction_normal) + [1] * len(prediction_anomaly)
        #values.append(roc_auc_score(true_labels, predictions))
        values.append(np.mean(prediction_anomaly > 0))
    return values

values = calculate_errors(all_gammas)
pl.plot(all_gammas, values)
pl.xscale('log')
pl.xlabel('Gamma')
pl.ylabel('Error on normal data (False Positive)')
pl.show()


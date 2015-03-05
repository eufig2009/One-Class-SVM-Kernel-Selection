import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.svm import OneClassSVM
from imbalanced import SMOTE
from pyDOE import lhs
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import pairwise_distances

ITERATION_COUNT = 50
DATA_SIZE = [1000]
DIMENSIONS = 20

def validate_classifier_by_random_points(clf, train_x, size=10000):
    clf.fit(train_x)
    positive_points = SMOTE(train_x, k=10,n_samples=size)
    negative_points = np.random.rand(size, train_x.shape[1]) - 0.5
    negative_points *= 10 * np.std(positive_points)
    error = (np.mean(clf.predict(positive_points) == -1) - clf.nu) ** 2
    return error

def parameters_selection_by_random_points(data, nu=0.05, weights=[0.99, 0.01], all_gammas = 2 ** np.arange(-10, 10, 50)):
    all_errors = []
    for index, gamma in enumerate(all_gammas):
        clf = OneClassSVM(nu=nu, gamma=gamma)
        err = validate_classifier_by_random_points(clf, data)
        all_errors.append(err)
    index = np.argmin(all_errors)
    return all_gammas[index], all_errors


def kernel_metric(data, gamma):
    distance = pairwise_distances(data)
    distance = distance * distance
    kernel_matrix = np.exp(-gamma * distance)
    kernel_matrix -= np.eye(kernel_matrix.shape[0])
    reguarization = 0.001
    return -np.var(kernel_matrix) / (np.mean(kernel_matrix) + reguarization)


def select_best_kernel_metric(data, all_gammas):
    all_errors = [kernel_metric(data, gamma) for gamma in all_gammas]
    index = np.argmin(all_errors)
    return all_gammas[index], all_errors

def select_best_support_vectors(data, nu=0.01, all_gammas=2 ** np.arange(-10, 10, 1)):
    all_errors = []
    for gamma in all_gammas:
        clf = OneClassSVM(nu=nu, gamma=gamma)
        clf.fit(data)
        prediction = clf.predict(data)
        out_of_class_count = np.sum(prediction == -1)
        support_vectors_count = len(clf.support_vectors_)
        error = (float(out_of_class_count) / len(data) - nu) ** 2
        error += (float(support_vectors_count) / len(data) - nu) ** 2
        all_errors.append(error)
    index = np.argmin(all_errors)
    return all_gammas[index], all_errors

def select_best_outlier_fraction_cross_val(data, nu=0.05, all_gammas=2 ** np.arange(-10, 10, 50), folds_count=7):
    all_errors = []
    kf_iterator = KFold(len(data), n_folds=folds_count)
    for gamma in all_gammas:
        error = 0
        for train, test in kf_iterator:
            train_data = data[train,:]
            test_data = data[test,:]
            clf = OneClassSVM(nu=nu, gamma=gamma)
            clf.fit(train_data)
            prediction = clf.predict(test_data)
            outlier_fraction = np.mean(prediction == -1)
            error += (nu - outlier_fraction) ** 2 + (float(clf.support_vectors_.shape[0]) / len(data) - nu) ** 2
        all_errors.append(error / folds_count)
    best_index = np.argmin(error)
    return int(best_index), all_errors

def slice_probability_space_selection(data, nu=0.05, all_gammas=2 ** np.linspace(-10, 10, 50),
    rho=0.05, outlier_distribution = np.random.rand, folds_count=7):
    kf_iterator = KFold(len(data), n_folds=folds_count)
    all_errors = []
    for gamma in all_gammas:
        error = 0.0
        clf = OneClassSVM(nu=nu, gamma=gamma)
        for train, test in kf_iterator:
            train_data = data[train,:]
            test_data = data[test,:]
            clf = OneClassSVM(nu=nu, gamma=gamma)
            clf.fit(train_data)
            prediction = clf.predict(test_data)
            inlier_metric_part = np.mean(prediction == -1)
            inlier_metric_part = inlier_metric_part / (1 + rho) / len(data)
            outliers = outlier_distribution(*data.shape) - 0.5
            outliers *= 8 * np.std(data)
            outlier_metric_part = np.mean(clf.predict(outliers) == 1) * rho / (1 + rho) / len(outliers)
            error += inlier_metric_part + outlier_metric_part
        all_errors.append(error / folds_count)
    index = np.argmin(all_errors)
    #best_index = pd.Series(all_errors).pct_change().argmax() - 1
    return int(index), all_errors


def work_pipe():
    all_gammas = 2 ** np.linspace(-20, 10, 50)

    slice_error_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    slice_error_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    outlier_fraction_error_cross_val_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    outlier_fraction_error_cross_val_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    outlier_fraction_error_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    outlier_fraction_error_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    kernel_error_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    kernel_error_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    kernel_error_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    kernel_error_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    random_generations_error_matrix = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))
    random_generations_error_std = np.zeros((len(all_gammas), len(range(2, DIMENSIONS, 2))))

    all_dimensions = range(2, DIMENSIONS, 2)
    for data_size in DATA_SIZE:

        for dim_index, dim in zip(range(len(all_gammas)), all_dimensions):
            print len(all_gammas)

            for gamma_index, gamma in enumerate(all_gammas):
                # Array initialization
                all_slice_error = np.zeros(ITERATION_COUNT)
                all_outlier_fraction_error_cross_val = np.zeros(ITERATION_COUNT)
                all_outlier_fraction_error = np.zeros(ITERATION_COUNT)
                all_kernel_error = np.zeros(ITERATION_COUNT)
                all_random_generations_error = np.zeros(ITERATION_COUNT)

                for iteration in xrange(ITERATION_COUNT):

                    data = np.random.randn(data_size, dim)


                    _, tmp = slice_probability_space_selection(data, all_gammas=np.array([gamma]))
                    all_slice_error[0] = tmp[0]
                    _, tmp = select_best_outlier_fraction_cross_val(data, all_gammas=np.array([gamma]))
                    all_outlier_fraction_error_cross_val[iteration] = tmp[0]
                    _, tmp = select_best_support_vectors(data, all_gammas=np.array([gamma]))
                    _, tmp = select_best_kernel_metric(data, all_gammas=np.array([gamma]))
                    all_kernel_error[iteration] = tmp[0]
                    _, tmp = parameters_selection_by_random_points(data, all_gammas=np.array([gamma]))
                    all_random_generations_error[iteration] = tmp[0]


                slice_error_matrix[gamma_index, dim_index] = np.mean(all_slice_error)
                slice_error_std[gamma_index, dim_index] = np.std(all_slice_error)
                outlier_fraction_error_cross_val_matrix[gamma_index, dim_index] = np.mean(all_outlier_fraction_error_cross_val)
                outlier_fraction_error_cross_val_std[gamma_index, dim_index] = np.std(all_outlier_fraction_error_cross_val)
                outlier_fraction_error_matrix[gamma_index, dim_index] = np.mean(all_outlier_fraction_error)
                outlier_fraction_error_std[gamma_index, dim_index] = np.std(all_outlier_fraction_error)
                kernel_error_matrix[gamma_index, dim_index] = np.mean(all_kernel_error)
                kernel_error_std[gamma_index, dim_index] = np.std(all_kernel_error)
                random_generations_error_matrix[gamma_index, dim_index] = np.mean(all_random_generations_error)

            np.savetxt('slice_error_dim{}_size{}'.format(dim, data_size), slice_error_matrix)
            np.savetxt('outlier_fraction_error_cross_val_matrix_dim{}_size{}'.format(dim, data_size), outlier_fraction_error_cross_val_matrix)
            np.savetxt('outlier_fraction_error_matrix_dim{}_size{}'.format(dim, data_size), outlier_fraction_error_matrix)
            np.savetxt('kernel_error_matrix_dim{}_size{}'.format(dim, data_size), kernel_error_matrix)
            np.savetxt('random_generations_error_matrix_dim{}_size{}'.format(dim, data_size), random_generations_error_matrix)


            np.savetxt('slice_std_dim{}_size{}'.format(dim, data_size), slice_error_std)
            np.savetxt('outlier_fraction_error_cross_val_std_dim{}_size{}'.format(dim, data_size), outlier_fraction_error_cross_val_std)
            np.savetxt('outlier_fraction_error_std_dim{}_size{}'.format(dim, data_size), outlier_fraction_error_std)
            np.savetxt('kernel_error_std_dim{}_size{}'.format(dim, data_size), kernel_error_std)
            np.savetxt('random_generations_error_std_dim{}_size{}'.format(dim, data_size), random_generations_error_std)


if __name__ == '__main__':
    work_pipe()

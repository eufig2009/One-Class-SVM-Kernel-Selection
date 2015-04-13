import pandas as pd
import sys
import os
from pylab import *

from sklearn.svm import SVDD
from sklearn.metrics import pairwise_distances, average_precision_score
from sklearn.cross_validation import KFold, train_test_split
from scipy.linalg import cholesky
from sklearn.utils import check_random_state
from scipy.spatial import cKDTree
from sklearn.utils import array2d
from random import sample

def SMOTE(X, n_samples, k=5, dist_power=2, multiple=False, sample_weight=None,
          random_state=None):
    """
    Returns n_samples of synthetic samples from X generated by SMOTE.

    Parameters
    ----------
    X : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    n_samples : int
        Number of new synthetic samples.
    k : int
        Number of nearest neighbours.
    dist_power : float, int
        Positive power in ditance metrics.
    random_state : None or int
        Seed for random generator.


    Returns
    -------
    smoted_X : array, shape = [n_samples, n_features]
        Synthetic samples
    """
    
    if type(X) is pd.core.frame.DataFrame:
        X = X.values

    rng = check_random_state(random_state)
    n_minor, n_features = X.shape
    k = min([n_minor - 1, k])

    # Learn nearest neighbours
    nn_tree = cKDTree(X)

    if multiple:
        smoted_X = X.copy()
        if sample_weight is not None:
            weight_smoted = sample_weight.copy()
        nn_dist, nn_idx = nn_tree.query(smoted_X, k=k + 1, p=dist_power)
        nn_idx = nn_idx[:, 1:]
        for i in xrange(n_samples):
            start_idx = rng.choice(len(smoted_X))
            start = smoted_X[start_idx, :]
            end_idx = nn_idx[start_idx, rng.choice(k)]
            end = smoted_X[end_idx, :]
            shift = rng.rand()
            new_point = [start * shift + end * (1. - shift)]
            new_nn_idx = np.argsort(
                distance_matrix(smoted_X, new_point,
                                p=dist_power).T)[::-1][0][:k]
            smoted_X = np.vstack((smoted_X, new_point))
            nn_idx = np.vstack((nn_idx, new_nn_idx))
            if sample_weight is not None:
                weight_smoted = \
                    np.concatenate((weight_smoted,
                                    weight_smoted[start_idx] * shift +
                                    (1. - shift) * weight_smoted[end_idx]))

    else:
        start_indices = rng.choice(len(X), size=(n_samples,))
        starts = X[start_indices, :]
        nn_dists, nn_idx = nn_tree.query(starts, k=k + 1, p=dist_power)
        end_indices = nn_idx[np.arange(n_samples),
                             rng.choice(np.arange(1, k + 1), n_samples)]
        ends = X[end_indices, :]
        shifts = rng.rand(n_samples)
        smoted_X = starts * np.repeat(array2d(shifts).T, n_features, axis=1) \
            + ends * np.repeat(array2d(1. - shifts).T, n_features, axis=1)
        smoted_X = np.vstack((X, smoted_X))
        if sample_weight is not None:
            weight_smoted = sample_weight[start_indices] * shifts\
                + (1. - shifts) * sample_weight[end_indices]
    if sample_weight is None:
        return smoted_X
    else:
        return smoted_X, np.concatenate((sample_weight, weight_smoted))

def generate_synthetic_data(size,dim=2, centroids=None, centroid_dispersion=1):
    size = int(size)
    all_data_part = []
    for point in centroids:
        covariation_matrix = np.random.randn(dim, dim)
        covariation_matrix = np.dot(covariation_matrix, covariation_matrix.T)
        covariation_matrix = cholesky(covariation_matrix) * centroid_dispersion
        data_part = np.random.randn(int(size / len(centroids)), dim)
        data_part = np.dot(data_part, covariation_matrix) + point
        all_data_part.append(data_part)
    data = np.concatenate(all_data_part, axis=0)
    return data

def generate_outlier(size, dim=2, space=1):
    size = int(size)
    outliers = (np.random.rand(size, dim) - 0.5) * space
    return outliers

def generate_dataset(size, outliers_part=0.12, dim=2, centroids=None, centroid_dispersion=1):
    inliers_part = 1.0 - outliers_part
    inliers = generate_synthetic_data(size * inliers_part, 
                                               dim, centroids, 
                                               centroid_dispersion)
    space = inliers.max() - inliers.min()
    outliers = generate_outlier(size*outliers_part, dim, space=space)
    data = np.concatenate([inliers, outliers], axis=0)
    data = pd.DataFrame(data)
    data['label'] = ['target'] * len(inliers) + ['outlier'] * len(outliers)
    return data


def slice_probability_metric(clf, train_x):
    nu = 1./clf.C/len(train_x)
    rho = nu / (1. - nu)
    clf.fit(train_x)
    positive_error = mean(clf.predict(train_x) == -1)
    positive_error /= (1 + rho)
    synthetic_anomalies = rand(1000, train_x.shape[1]) - 0.5
    synthetic_anomalies *= 4 * (train_x.max() - train_x.min())
    negative_errors = mean(clf.predict(synthetic_anomalies) == 1)
    negative_errors *= rho/(1 + rho)
    return negative_errors + positive_error


def validate_classifier_by_random_points(clf, train_x, size=10000):
    nu = 1./clf.C/len(train_x)
    rho = nu / (1. - nu)
    clf.fit(train_x)
    positive_points = SMOTE(train_x, k=5,n_samples=size)
    negative_points = np.random.rand(size, train_x.shape[1])
    negative_points *= 4 * (train_x.max() - train_x.min())
    error = mean(clf.predict(positive_points) == -1) / (1 + rho)
    error += mean(clf.predict(negative_points) == 1) * rho/(1 + rho)
    return error

def rbf_kernel(X, Y, gamma):
    distance_matrix = pairwise_distances(X, Y) ** 2
    distance_matrix = exp(-1 * gamma * distance_matrix)
    return distance_matrix

def calculate_radius(clf):
    all_support_vectors = clf.support_vectors_
    first_support_vector = clf.support_vectors_[0, :]
    dual_coef = clf.dual_coef_
    gamma = clf.gamma
    test_vector_norm = 1
    second_part = rbf_kernel(all_support_vectors, first_support_vector, gamma)
    #print "second_vector", second_part
    second_part = dot(dual_coef, second_part)
    #print second_part
    third_part = dot(dual_coef, dot(rbf_kernel(all_support_vectors, all_support_vectors, gamma), dual_coef.T))[0, 0]
    radius = (test_vector_norm - 2 * second_part + third_part)
    return radius[0, 0]

def combinatorial_dimension_metric(clf, train_x):
    clf.fit(train_x)
    prediction = clf.decision_function(train_x)
    negative_marks = prediction < 0
    prediction = prediction[negative_marks]
    distance = prediction.min()
    radius = calculate_radius(clf)
    radius += abs(distance)
    return abs(distance) / radius

def kernel_metric(clf, train_x):
    gamma = clf.gamma
    distance = pairwise_distances(train_x)
    distance = distance * distance
    kernel_matrix = exp(-gamma * distance)
    kernel_matrix -= eye(kernel_matrix.shape[0])
    reguarization = 0.001
    return -var(kernel_matrix) / (mean(kernel_matrix) + reguarization)

def support_vectors_metric(clf, train_x):
    nu = 1./ clf.C / len(train_x)
    clf.fit(train_x)
    prediction = clf.predict(train_x)
    out_of_class_fraction = mean(prediction == -1)
    support_vectors_fraction = float(len(clf.support_vectors_)) / len(train_x)
    metric = (out_of_class_fraction - nu) ** 2
    metric += (support_vectors_fraction - nu) ** 2
    return metric

def nu_variation_criteria(model, train_x, folds_count=10):
    all_nu = []
    kfolds = KFold(n=len(train_x), n_folds=folds_count, shuffle=True)
    C = model.C
    nu = 1./C / len(train_x)
    for train, test in kfolds:
        model.fit(train_x[train, :])
        predictions = model.predict(train_x[test,:])
        all_nu.append((mean(predictions == -1) - nu)**2)
    return mean(all_nu)


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
    kfold =KFold(len(data), n_folds=parts_count, shuffle=True)
    for train, test in kfold:
        all_parts.append(data.iloc[test, :])
    return all_parts


def validate_gamma(train, test_normal, test_anomaly, gamma):
    C = 1.0 / (0.1 * (len(train)))
    clf = SVDD(kernel='rbf', gamma=gamma, C=C)
    #clf.fit(np.random.randn(10000, 4))
    clf.fit(train)
    normal_data_prediction = clf.decision_function(test_normal)
    anomaly_data_prediction = clf.decision_function(test_anomaly)
    normal_data_error = np.mean(normal_data_prediction < 0)
    anomaly_data_error = np.mean(anomaly_data_prediction > 0)
    true_labels = [1] * len(test_normal) + [-1] * len(test_anomaly)
    decision_values = np.concatenate([normal_data_prediction, anomaly_data_prediction], axis=0)
    precision, recall, _ = precision_recall_curve(true_labels, decision_values)
    auc_score = auc(recall, precision)
    return normal_data_error, anomaly_data_error, auc_score


def single_experiment_false_fraction(data, gamma, nu):
    C = 1./ len(data) / nu
    model = SVDD(kernel='rbf', C=C, gamma=gamma)
    normal_data, anomaly_data = split_anomaly_normal_data(data)
    anomaly_elements_count = int(len(normal_data) * nu / (1. - nu))
    rows = sample(anomaly_data.index, anomaly_elements_count)
    anomaly_data = anomaly_data.ix[rows]
    normal_train, normal_validate, normal_test = split_data_set(normal_data, 3)
    anomaly_train, anomaly_validate, anomaly_test = split_data_set(anomaly_data, 3)
    anomaly_train = concatenate([anomaly_train, anomaly_validate])
    normal_train = concatenate([normal_train, normal_validate])
    model.fit(np.concatenate([anomaly_train, normal_train]))
    anomaly_prediction = model.decision_function(anomaly_test)
    normal_prediction = model.decision_function(normal_test)
    false_anomaly = mean(normal_prediction < 0)
    false_normal = mean(anomaly_prediction > 0)
    prediction = concatenate([anomaly_prediction, normal_prediction])
    true_labels = array([1] * len(anomaly_prediction) + [-1] * len(normal_prediction))
    auc_score = average_precision_score(true_labels, -1 * prediction)
    train_data = concatenate([anomaly_train, normal_train])
    slice_score = slice_probability_metric(model, train_data)
    support_score = support_vectors_metric(model, train_data, nu)
    smote_score = validate_classifier_by_random_points(model, train_data, (1. - nu)/nu)
    vc_score = combinatorial_dimension_metric(model, train_data)
    kernel_score = kernel_metric(model, train_data)
    return false_anomaly, false_normal, auc_score, \
           slice_score, smote_score, vc_score, support_score, kernel_score


def generate_anomalies(data, anomaly_fraction=0.1):
    fraction = anomaly_fraction / (1. - anomaly_fraction)
    anomaly_count = int(len(data) * anomaly_fraction)
    anomaly = rand(anomaly_count, data.shape[1])
    anomaly -= 0.5
    anomaly *= (data.max(axis=0) - data.min(axis=0))[newaxis, :]
    anomaly += data.mean(axis=0)[newaxis, :]
    anomaly = pd.DataFrame(anomaly)
    return anomaly

def first_experiment(data, model, anomaly_fraction=0.1):
    risk_functions = [support_vectors_metric, kernel_metric, validate_classifier_by_random_points,
    combinatorial_dimension_metric, slice_probability_metric]
    train, test = train_test_split(data, train_size=0.7)
    model.fit(train)
    anomaly_test = generate_anomalies(test, anomaly_fraction)
    anomaly_prediction = model.predict(anomaly_test)
    normal_prediction = model.predict(test)
    false_anomaly = mean(normal_prediction == -1)
    false_normal = mean(anomaly_prediction == 1)
    results = []
    for metric in risk_functions:
        results.append(metric(model, train))
    return false_normal, false_anomaly, results

def second_experiment(data, model, anomaly_fraction):
    risk_functions = [support_vectors_metric, kernel_metric, validate_classifier_by_random_points,
    combinatorial_dimension_metric, slice_probability_metric]
    anomalies = generate_anomalies(data, anomaly_fraction)
    model.fit(concatenate([data.values, anomalies]))
    anomaly_prediction = model.predict(anomalies)
    normal_prediction = model.predict(data)
    false_normal = mean(anomaly_prediction == 1)
    false_anomaly = mean(normal_prediction == -1)
    results = []
    for metric in risk_functions:
        results.append(metric(model, data))
    return false_normal, false_anomaly, results


def check_all(data_file):
    data = pd.read_csv('./csv_data_set/{}'.format(data_file))
    data = data.query("label == 'target'").drop(['label'], axis=1)
    all_anomaly_fraction = [0.01, 0.05, 0.1, 0.2, 0.3]
    all_nu = [0.01, 0.05, 0.1, 0.2, 0.3]
    all_gammas = logspace(-10, 10, 50)
    final_shape = (5, 5, 50)
    all_false_anomaly = zeros(final_shape)
    all_false_normal = zeros(final_shape)
    all_kernel = zeros(final_shape)
    all_support = zeros(final_shape)
    all_smote = zeros(final_shape)
    all_emperic = zeros(final_shape)
    all_vc = zeros(final_shape)
    for fraction_index, fraction in enumerate(all_anomaly_fraction):
        for nu_index, nu in enumerate(all_nu):
            for gamma_index, gamma in enumerate(all_gammas):
            	C = 1./len(data)/nu
                model = SVDD(kernel='rbf', gamma=gamma, C=C)
                all_false_normal[fraction_index, nu_index, gamma_index], \
                all_false_anomaly[fraction_index, nu_index, gamma_index],\
                tmp = first_experiment(data, model, fraction)

                all_support[fraction_index, nu_index, gamma_index] = tmp[0]
                all_kernel[fraction_index, nu_index, gamma_index] = tmp[1]
                all_smote[fraction_index, nu_index, gamma_index] = tmp[2]
                all_vc[fraction_index, nu_index, gamma_index] = tmp[3]
                all_emperic[fraction_index, nu_index, gamma_index] = tmp[4]

    name = data_file[:-4]
    save('{}_false_normal.csv'.format(name), all_false_normal)
    save('{}_false_anomaly.csv'.format(name), all_false_anomaly)
    save(arr=all_support, file='{}_support.csv'.format(name))
    save(arr=all_emperic, file='{}_all_emperic.csv'.format(name))
    save(arr=all_smote, file='{}_all_smote.csv'.format(name))
    save(arr=all_kernel, file='{}_all_kernel.csv'.format(name))
    save(arr=all_vc, file='{}_all_vc.csv'.format(name))


def check_all_second(data_file):
    data = pd.read_csv('./csv_data_set/{}'.format(data_file))
    data = data.query("label == 'target'").drop(['label'], axis=1)
    all_anomaly_fraction = [0.01, 0.05, 0.1, 0.2, 0.3]
    all_nu = [0.01, 0.05, 0.1, 0.2, 0.3]
    all_gammas = logspace(-10, 10, 50)
    final_shape = (5, 5, 50)
    all_false_anomaly = zeros(final_shape)
    all_false_normal = zeros(final_shape)
    all_kernel = zeros(final_shape)
    all_support = zeros(final_shape)
    all_smote = zeros(final_shape)
    all_emperic = zeros(final_shape)
    all_vc = zeros(final_shape)
    for fraction_index, fraction in enumerate(all_anomaly_fraction):
        for nu_index, nu in enumerate(all_nu):
            for gamma_index, gamma in enumerate(all_gammas):
            	C = 1./len(data)/nu
                model = SVDD(kernel='rbf', gamma=gamma, C=C)
                all_false_normal[fraction_index, nu_index, gamma_index], \
                all_false_anomaly[fraction_index, nu_index, gamma_index],\
                tmp = second_experiment(data, model, fraction)

                all_support[fraction_index, nu_index, gamma_index] = tmp[0]
                all_kernel[fraction_index, nu_index, gamma_index] = tmp[1]
                all_smote[fraction_index, nu_index, gamma_index] = tmp[2]
                all_vc[fraction_index, nu_index, gamma_index] = tmp[3]
                all_emperic[fraction_index, nu_index, gamma_index] = tmp[4]

    name = data_file[:-4]
    save('{}_false_normal_second.csv'.format(name), all_false_normal)
    save('{}_false_anomaly_second.csv'.format(name), all_false_anomaly)
    save(arr=all_support, file='{}_support_second.csv'.format(name))
    save(arr=all_emperic, file='{}_all_emperic_second.csv'.format(name))
    save(arr=all_smote, file='{}_all_smote_second.csv'.format(name))
    save(arr=all_kernel, file='{}_all_kernel_second.csv'.format(name))
    save(arr=all_vc, file='{}_all_vc_second.csv'.format(name))

if __name__ == '__main__':
    log_file = open('res.log', 'w')
    all_files = os.listdir('./csv_data_set')
    for index, file_name in enumerate(all_files):
        print 'processed {}'.format(index), '{} left'.format(len(all_files) - index)
        try:
            check_all(file_name)
            check_all_second(file_name)
            log_file.write('{} is finished\n'.format(file_name))
        except Exception, exception_text:
            log_file.write('{}\n'.format(exception_text))


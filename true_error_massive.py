import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cross_validation import train_test_split

nu = 0.1
features_count = 20
clf = OneClassSVM(nu=nu)
all_gammas = 2 ** np.linspace(-10, 10, 20)

iterations = 20
#data = make_classification(n_samples=100000, n_features=features_count,n_redundant=0, n_classes=2, weights=[0.00])[0]
results_positive = np.zeros((len(all_gammas), len(range(2, features_count, 2))))
results_negative = np.zeros((len(all_gammas), len(range(2, features_count, 2))))
for dim_index, dim in enumerate(range(2, features_count, 2)):

    for iteration in xrange(iterations):
        data = np.random.randn(10000, features_count)
        outlier_count = int(len(data) * nu)
        #second_class = (lhs(data.shape[1], samples=outlier_count)) * (data.max().max() - data.min().min()) * 3
        #second_class -= mean(second_class)
        second_class = np.random.rand(*data.shape) - 0.5
        second_class *= 10
        labels = np.array([1] * len(data) + [-1] * len(second_class))
        data = np.r_[data, second_class]
        train, test, train_y, test_y = train_test_split(data, labels, train_size=0.5)
        all_errors_positive = []
        all_errors_negative = []
        all_errors = []
        for index, gamma in enumerate(all_gammas):
            clf.gamma = gamma
            clf.fit(train)
            inliers_marks = test_y == 1
            outlier_marks = test_y == -1
            inliers_marks = inliers_marks.nonzero()[0]
            outlier_marks = outlier_marks.nonzero()[0]
            test_iniers = test[inliers_marks, :]
            test_outliers = test[outlier_marks, :]
            prediction = clf.predict(test_iniers)
            all_errors_positive.append(np.mean(prediction == -1))
            prediction = clf.predict(test_outliers)
            all_errors_negative.append(np.mean(prediction == 1))
        print all_errors_negative, all_errors_positive
        results_positive[:, dim_index] += all_errors_positive
        print results_positive
        results_negative[:, dim_index] += all_errors_negative
        print results_negative


np.savetxt('positive_error', results_positive)
np.savetxt('negative_error', results_negative)





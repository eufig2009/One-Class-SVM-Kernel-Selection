import numpy as np
import pylab as pl
from sklearn.svm import OneClassSVM


data = np.random.randn(100, 2)

x = np.arange(-5, 5,  0.1)
y = np.arange(-5, 5, 0.1)

XX, YY = np.meshgrid(x, y)


clf = OneClassSVM(nu=0.1, gamma=0.01)
clf.fit(data)

Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
pl.contourf(XX, YY, Z > 0, cmap = 'gist_earth')
prediction = clf.predict(data)
positive_exaples = prediction == 1
positive_exaples = positive_exaples.nonzero()[0]
negative_example = prediction == -1
negative_example = negative_example.nonzero()[0]
pl.scatter(data[positive_exaples, 0], data[positive_exaples, 1], facecolors='none', zorder=10)
pl.scatter(data[negative_example, 0], data[negative_example, 1], facecolors = 'red', zorder=10, label='Anomaly')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

pl.rc('font', **font)
pl.title('gamma = 0.01')
pl.legend(loc='best')
pl.show()

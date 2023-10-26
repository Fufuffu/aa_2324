import numpy
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler


def kernel_lineal(x, z):
    return x.dot(z.T)


def kernel_gaussia(x, z, gamma=10):
    return np.exp(-gamma * np.power(distance_matrix(x, z), 2))


def kernel_polynomial(x, z, gamma=10, degree=2, coef=0):
    return np.power(gamma * x.dot(z.T) + coef, degree)


X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# En realitat ja no necessitem canviar les etiquetes Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)


def prova_kernels_lineals():
    svc_scikit = SVC(C=10000, kernel='linear')
    svc_scikit.fit(X_transformed, y_train)
    svc_prediction_scikit = svc_scikit.predict(X_test_transformed)

    svc = SVC(C=10000, kernel=kernel_lineal)
    svc.fit(X_transformed, y_train)
    svc_prediction = svc.predict(X_test_transformed)

    print(numpy.all(svc_prediction_scikit == svc_prediction))


def prova_kernels_gaussia():
    svc_scikit = SVC(C=100, kernel='rbf', gamma=10)
    svc_scikit.fit(X_transformed, y_train)
    svc_prediction_scikit = svc_scikit.predict(X_test_transformed)

    svc = SVC(C=100, kernel=kernel_gaussia)
    svc.fit(X_transformed, y_train)
    svc_prediction = svc.predict(X_test_transformed)

    print(precision_score(y_test, svc_prediction_scikit))
    print(precision_score(y_test, svc_prediction))


def prova_kernels_polinomial():
    svc_scikit = SVC(C=100, kernel='poly', gamma=10, degree=2)
    svc_scikit.fit(X_transformed, y_train)
    svc_prediction_scikit = svc_scikit.predict(X_test_transformed)

    svc = SVC(C=100, kernel=kernel_polynomial)
    svc.fit(X_transformed, y_train)
    svc_prediction = svc.predict(X_test_transformed)

    print(precision_score(y_test, svc_prediction_scikit))
    print(precision_score(y_test, svc_prediction))


prova_kernels_polinomial()

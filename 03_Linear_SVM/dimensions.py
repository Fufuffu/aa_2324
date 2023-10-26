import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)

# Entrenam una SVM linear (classe SVC)

svc = SVC(C=10000, kernel='linear')
svc.fit(X_transformed, y_train)

# Prediccio
svc_prediction = svc.predict(X_test)

# Metrica
correct_predictions = np.count_nonzero((svc_prediction - y_test) == 0)
print(correct_predictions / y_test.shape[0])

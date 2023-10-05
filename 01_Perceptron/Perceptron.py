import numpy as np


class Perceptron:

    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting. w[0] = threshold
    errors_ : list
        Number of miss classifications in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None  # defined in method fit

    def fit(self, X, y):

        """Fit training dat.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        """
        self.w_ = np.zeros(1 + X.shape[1])  # First position corresponds to threshold

        for _ in range(self.n_iter):
            for x_row, y_row in zip(X, y):
                y_predicted = self.__prediction_value(x_row)
                weight_row_tuple = zip(self.w_, np.append([1], x_row))
                self.w_ = [w_j + self.eta * (y_row - y_predicted) * x_j for w_j, x_j in weight_row_tuple]

    def predict(self, X):
        """Return class label.
            First calculate the output: (X * weights) + threshold
            Second apply the step function
            Return a list with classes
        """

        return [self.__prediction_value(x_row) for x_row in X]

    def __prediction_value(self, x_row):
        predicted = self.w_[0] + x_row[0] * self.w_[1] + x_row[1] * self.w_[2]
        return -1 if predicted <= 0 else 1

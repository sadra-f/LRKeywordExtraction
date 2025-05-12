import numpy as np
import pandas as pd
class LogisticRegression:
    def __init__(self, epochs:int=30000, learning_rate:float=0.01):
        self.weights = None
        self.epochs= epochs
        self.learning_rate = learning_rate
    def _standardize(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    def fit(self, X, Y):
        #TODO: standardize and save standanardized values for later use..
        X = np.array(X)
        Y = np.array(self._Check_binary_classes(Y))
        self._initiate_weights(len(X[0]))
        for i in range(self.epochs):
            predictions = self._sigmoid(np.sum(np.multiply(X, self.weights), axis=1))
            loss = -(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
            tloss = np.sum(loss)
            weights = weights - (self.learning_rate / len(X)) * np.dot(X.T, (predictions - Y))
    def _initiate_weights(self, size):
        self.weights = np.random.random_sample(size)
    def predict(self, data):
        pass
    def save_weights(self, path):
        pass
    def load_weights(self, path):
        pass
    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    def _Check_binary_classes(self, data:pd.DataFrame):
        if data.dtypes == bool:
            data = data.astype(int)
        return data
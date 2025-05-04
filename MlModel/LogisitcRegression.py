import numpy as np
class LogisticRegression:
    def __init__(self):
        self.weights = None
    def _standardize(self, data):
        return data - np.mean(data, axis=0) / np.std(data, axis=0)
    def fit(self, data):
        pass
    def predict(self, data):
        pass
    def save_weights(self, path):
        pass
    def load_weights(self, path):
        pass
    def _sigmoid(self, data):
        return 1 / 1 + np.exp(data)
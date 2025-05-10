import numpy as np
class LogisticRegression:
    def __init__(self, steps:int=10000):
        self.weights = None
        self.steps= steps
    def _standardize(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    def fit(self, data):
        self._initiate_weights()
        for i in range(self.steps):
            results = self._sigmoid(np.multiply(data, self.weights))
    def _initiate_weights(self):
        self.weights = np.random.random_sample()
    def predict(self, data):
        pass
    def save_weights(self, path):
        pass
    def load_weights(self, path):
        pass
    def _sigmoid(self, data):
        return 1 / (1 + np.exp(data))
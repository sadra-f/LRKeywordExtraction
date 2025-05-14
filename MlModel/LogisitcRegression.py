import numpy as np
import pandas as pd
import pickle as pkl
from matplotlib import pyplot


class LogisticRegression:
    def __init__(self, epochs:int=30000, learning_rate:float=0.01, keep_loss_hist=True):
        self.weights = None
        self.epochs= epochs
        self.learning_rate = learning_rate
        self.keep_loss_hist = keep_loss_hist
        self.loss_hist = []

    def _standardize(self, data):
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0), np.mean(data, axis=0), np.std(data, axis=0)
    # TODO L fix standardization zero division
    def fit(self, X, Y):
        Y = np.array(self._Check_binary_classes(Y))
        X, self.X_mean, self.X_std = self._standardize(np.array(X))

        self._initiate_weights(len(X[0]))

        for i in range(self.epochs):
            predictions = self._sigmoid(np.sum(np.multiply(X, self.weights), axis=1))
            loss = -(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
            tloss = np.sum(loss)
            if self.keep_loss_hist and i%100 == 0:
                self.loss_hist.append(tloss)

            self.weights = self.weights - (self.learning_rate / len(X)) * np.dot(X.T, (predictions - Y))
        if self.loss_hist:
            pyplot.figure()
            pyplot.plot([i for i in range(len(self.loss_hist))], self.loss_hist)
        
        return

    def _initiate_weights(self, size):
        self.weights = np.random.random_sample(size)

    def predict(self, X):
        X = (X - self.X_mean) / self.X_std
        self._sigmoid(np.sum(np.multiply(X, self.weights), axis=1))

    def save_model(self, path="log/LR_model.pkl"):
        value = {"X_MEAN":self.X_mean, "X_STD":self.X_std, "W":self.weights}
        with open(path, "wb") as f:
            pkl.dump(value, f)
    
    def load_weights(self, path="log/LR_model.pkl"):
        value = None
        with open(path, "rb") as f:
            value = pkl.load(f)
        self.X_std = value["X_STD"]
        self.weights = value["W"]
        self.X_mean = value["X_MEAN"]

    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    
    def _Check_binary_classes(self, data:pd.DataFrame):
        if data.dtypes == bool:
            data = data.astype(int)
        return data
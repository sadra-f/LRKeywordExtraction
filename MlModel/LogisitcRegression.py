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
        self._loss_rec_steps = 100
        self._ref_vals = None

    def _standardize(self, data, exclude):
        """Applies Standarization to the data

        Args:
            data (_type_): data to standarize
            exclude (_type_): columns to exclude from standardization

        Returns:
            list[list], dict: returns the standarized dataset along mean and std for 
            each column to be applied to prediction data as to fit well to the weights
        """
        ref_vals = []
        for i in range(len(data[0])):
            if i in exclude:
                ref_vals.append((i, None, None))
            else:
                ref_vals.append((i, np.mean(data[:, i], axis=0), np.std(data[:, i], axis=0)))
                data[:,i] =  (data[:, i] - ref_vals[i][1]) / ref_vals[i][2]
        return data, ref_vals
    
    def fit(self, X, Y, standardization:bool=True, onehot_col_index:list[int]=None):
        """Fits the model weightd to the input data

        Args:
            X (_type_): Input feature vectors
            Y (_type_): input class vector
            standardization (bool, optional): If applying standardization is required. Defaults to True.
            onehot_col_index (list[int], optional): Index of one-hot encoded columns as to not apply standardization to them. Defaults to None.
        """
        Y = np.array(self._Check_binary_classes(Y))
        if standardization:
            X, self._ref_vals = self._standardize(np.array(X), onehot_col_index)
        else:
            X = np.array(X)

        self._initiate_weights(len(X[0]))

        for i in range(self.epochs):
            predictions = self._sigmoid(np.dot(X, self.weights[1:]) + self.weights[0])
            loss = -(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
            tloss = np.sum(loss)
            if self.keep_loss_hist and i%self._loss_rec_steps == 0:
                self.loss_hist.append(tloss)

            self.weights[1:] = self.weights[1:] - (self.learning_rate / len(X)) * np.dot(X.T, (predictions - Y))
            self.weights[0] = self.weights[0] - (self.learning_rate / len(X)) * np.sum((predictions - Y))
        if self.keep_loss_hist:
            pyplot.figure()
            pyplot.plot([i*self._loss_rec_steps for i in range(len(self.loss_hist))], self.loss_hist)
            pyplot.ylabel("Loss Value")
            pyplot.xlabel("Iteration")
            pyplot.title("Loss over Iterations")

        
        return

    def _initiate_weights(self, size):
        # +1 for the bias
        self.weights = np.random.random_sample(size + 1)

    def _apply_standardization(self, data):
        """Applies existing mean and std values to new data

        Args:
            data (_type_): input dataset

        Raises:
            Exception: if no logged mean and std values exist

        Returns:
            _type_: standardized dataset
        """
        if self._ref_vals is None:
            raise Exception("Standardization was not set for this object")
        data = np.array(data)
        for i, mean, std in self._ref_vals:
            if mean != None:
                data[:, i] = (data[:, i] - mean) / std
        return data

    def predict(self, X):
        if self._ref_vals != None:
            X = self._apply_standardization(X)
        predicts = self._sigmoid(np.dot(X, self.weights[1:]) + self.weights[0])
        return predicts > 0.6
        

    def save_model(self, path="log/LR_model.pkl"):
        value = {"ref_vals":self._ref_vals, "W":self.weights}
        with open(path, "wb") as f:
            pkl.dump(value, f)
    
    def load_model(self, path="log/LR_model.pkl"):
        value = None
        with open(path, "rb") as f:
            value = pkl.load(f)
        self._ref_vals = value["ref_vals"]
        self.weights = value["W"]

    def _sigmoid(self, data):
        return 1 / (1 + np.exp(-data))
    
    def _Check_binary_classes(self, data:pd.DataFrame):
        if data.dtypes == bool:
            data = data.astype(int)
        return data
from Visualize.Dataset import plot
import pandas as pd
from MlModel.LogisitcRegression import LogisticRegression as LR
import numpy as np
from Evaluation.ClassificationEvaluators import evaluate_classification as ec

# use this sample to train the Logistic Regression model with provided dataset

dataset = pd.read_csv("dataset/numerical_dataset.csv")
dataset.pop("word")

t = np.random.rand(len(dataset)) < 0.75
train = dataset[t]
test = dataset[~t]

Y_train = train.pop(dataset.columns[-1])
X_train = train

Y_test = test.pop(test.columns[-1])
X_test = test

lr = LR()

lr.fit(X_train, Y_train, onehot_col_index=[i for i in range(13, 48, 1)])
lr.save_model()

lr.load_model()
Y_predict = lr.predict(X_test)

print(ec(Y_predict, Y_test))
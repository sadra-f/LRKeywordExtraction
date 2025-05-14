from Visualize.Dataset import plot
import pandas as pd
from MlModel.LogisitcRegression import LogisticRegression as LR
import numpy as np
from Evaluation.ClassificationEvaluators import evaluate_classification as ec


dataset = pd.read_csv("dataset/numerical_dataset.csv")
dataset.pop("word")
t = np.random.rand(len(dataset)) < 0.75
train = dataset[t]
test = dataset[~t]
Y_train = train.pop(dataset.columns[-1])
X_train = train
Y_test = test.pop(test.columns[-1])
X_test = test
# plot(dataset.iloc[:, 0:13], Y)
lr = LR()
lr.load_weights("log/LR_model.pkl")
# lr.fit(X_train, Y_train, [i for i in range(13, 48, 1)])
Y_predict = lr.predict(X_test)
print(ec(Y_predict, Y_test))
print()
from Visualize.Dataset import plot
import pandas as pd
from MlModel.LogisitcRegression import LogisticRegression as LR
import numpy as np
from Evaluation.ClassificationEvaluators import evaluate_classification as ec
from matplotlib import pyplot as plt

''' use this sample to train the Logistic Regression model with provided dataset '''

# Load dataset
dataset = pd.read_csv("dataset/numerical_dataset.csv")
dataset.pop("word")
columns_to_plot = dataset.iloc[:, 1:13]

# Iterate over the selected columns and plot each one
for col_name in columns_to_plot.columns:
    plt.figure(figsize=(10, 5))  # Create a new figure for each column
    trues = columns_to_plot[dataset["keyword"]==True][col_name]
    falses = columns_to_plot[dataset["keyword"]==False][col_name]
    plt.scatter(trues, [i for i in range(len(trues))], color='blue')
    plt.scatter(falses, [i for i in range(len(falses))], color='red')
    plt.title(f'Values of {col_name} over Record Number')
    plt.xlabel('Record Number')
    plt.ylabel(col_name)
    plt.grid(True)
    plt.show()
# Split dataset to train and test
t = np.random.rand(len(dataset)) < 0.75
train = dataset[t]
test = dataset[~t]

Y_train = train.pop(dataset.columns[-1])
X_train = train

Y_test = test.pop(test.columns[-1])
X_test = test
# Build Logistic Regression models object
lr = LR()
# Fit the model to the data
lr.fit(X_train, Y_train, onehot_col_index=[i for i in range(13, 48, 1)])
# Save the model onto a file
lr.save_model("log/LR_model.pkl")
# Load data from file (saving and then loading is for exemplar purposes)
lr.load_model("log/LR_model.pkl")
# Predict for the test set
Y_predict = lr.predict(X_test)
# Print Evaluation measures
print(ec(Y_predict, Y_test))
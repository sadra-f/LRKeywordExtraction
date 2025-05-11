from Visualize.Dataset import plot
import pandas as pd

dataset = pd.read_csv("dataset/numerical_dataset.csv")
Y = dataset.pop(dataset.columns[-1])
dataset.pop("word")
plot(dataset.iloc[:, 0:13], Y)
print()
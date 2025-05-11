from matplotlib import pyplot as plt

def plot(X, Y):
    fig, axs = plt.subplots(3, 5)
    for i, col_name in enumerate(X.columns):
        axs[i//5, i%5].scatter(X[col_name], Y)
        axs[i//5, i%5].set_title(col_name)
    fig.show()
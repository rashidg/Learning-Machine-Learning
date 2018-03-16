from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from matplotlib import markers
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names

    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        splot = plt.subplot(3, 5, i + 1)
        # Plot feature i against y
        x = [x[i] for x in X]
        splot.plot(x, y, '.')
        splot.set_title(features[i])

    plt.tight_layout()
    plt.show()


def predict(X, w):
    ones = np.ones((len(X), 1))
    X1 = np.hstack((ones, X))
    return X1.dot(w)


def fit_regression(X,Y):
    ones = np.ones((len(X), 1))
    X = np.hstack((ones, X))

    # ax = b
    # x: w
    # a: X^T * X
    # b: X^T * Y
    a = np.dot(X.transpose(), X)
    b = np.dot(X.transpose(), Y)
    w = np.linalg.solve(a, b)

    return w


def graph(y, y_p):
    plt.figure(figsize=(15, 8))
    splot = plt.subplot(1,1,1)
    splot.scatter(range(len(y)), y, marker='.')
    splot.scatter(range(len(y)), y_p, marker='.', color='red')
    plt.tight_layout()
    plt.show()


def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # Split data into train and test
    idx = np.random.choice(506, 506, replace=False)
    test_len = len(X) // 5
    trn_len = len(X) - test_len
    trn_X = X[idx[:trn_len]]
    trn_y = y[idx[:trn_len]]
    test_X = X[idx[trn_len:]]
    test_y = y[idx[trn_len:]]
    
    # Fit regression model
    w = fit_regression(trn_X, trn_y)
    # Compute fitted values, MSE, etc.
    pred = predict(test_X, w)
    print("mean squared error: ", np.mean((pred - test_y)**2))
    print("root mean squared error: ", np.sqrt(np.mean((pred - test_y)**2)))
    print("sum of squared errors: ", np.sum((pred-test_y)**2))
    
    # graph(test_y, pred)


if __name__ == "__main__":
    main()


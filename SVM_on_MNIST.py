import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''

    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return the updated parameters
        # grad: vector of the same size as the number of parameters
        self.vel = self.beta * self.vel - self.lr * grad
        #params += self.vel
        return params + self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.b = np.random.normal(0.0, 0.1, 1)
        self.supports = []

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.

        if hinge loss > 1:
        -means t_i(wTx_i + b) <0: wrongly classified,
        so data point x_i is on the other side of threshold line (i.e.y=0 line)

        if 0 < hinge loss < 1:
        -means 0 < t_i(wTx_i + b) < 1: data point x_i is correctly classified
        but is even closer to the threshold line (i.e. y=0) than the support vector points (where  t_i(wTx_i + b) = 1)


        to figure out the support vectors (non-zero loss)
        '''
        loss = []

        for i in range(len(X)):
            x = X[i]
            t = y[i]
            loss.append(max(0, 1 - t * (np.dot(np.transpose(self.w), x) + self.b)))

        loss = np.array(loss)
        return loss

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).

        uses hinge_loss

        '''
        # Compute (sub-)gradient of SVM objective
        losses = self.hinge_loss(X, y)

        subgradient = 0
        b_gradient = 0
        for i in range(len(X)):
            if losses[i] != 0:
                subgradient += np.dot(y[i], X[i])
                b_gradient += y[i]

        return (self.w - self.c * subgradient, -self.c * b_gradient)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))

        y = sign(b + wTx)
        '''
        # Classify points as +1 or -1
        pred = []

        for i in range(len(X)):
            x = X[i]
            if np.dot(np.transpose(self.w), x) + self.b < 0:
                pred.append(-1)
            else:
                pred.append(1)

        pred = np.array(pred)
        return pred

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        w = optimizer.update_params(w, func_grad(w))
        w_history.append(w)

    return w_history

def test_and_visualize_gd():
    no_momentum = optimize_test_function(GDOptimizer(1.0, 0.0))
    yes_momentum = optimize_test_function(GDOptimizer(1.0, 0.9))
    x = np.arange(201)
    plt.plot(x, no_momentum)
    plt.plot(x, yes_momentum)
    plt.legend(['without momentum', 'with momentum'], loc='upper center')
    plt.show()

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters, lr, momentum):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    opt = optimizer(lr, momentum)
    svm = SVM(penalty, train_data.shape[1])
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    for _ in range(iters):
        X_batch, y_batch = batch_sampler.get_batch()
        (gradient, b_gradient) = svm.grad(X_batch, y_batch)

        new_params = opt.update_params(np.append(svm.w, svm.b), np.append(gradient, b_gradient))
        svm.w = new_params[:-1]
        svm.b = new_params[-1]

    return svm

def reports(svm):
    train_loss = svm.hinge_loss(train_data, train_targets)
    print("mean train loss: ", np.mean(train_loss))
    test_loss = svm.hinge_loss(test_data, test_targets)
    print("mean test loss: ", np.mean(test_loss))

    pred = svm.classify(train_data)
    hit = 0
    for i in range(len(train_data)):
        if train_targets[i] == pred[i]:
            hit += 1
    print("train accuracy: ", hit / len(train_data))

    pred = svm.classify(test_data)
    hit = 0
    for i in range(len(test_data)):
        if test_targets[i] == pred[i]:
            hit += 1
    print("test accuracy: ", hit / len(test_data))

    plt.imshow(np.split(svm.w, 28), cmap='gray')
    plt.show()

if __name__ == '__main__':
    train_data, train_targets, test_data, test_targets = load_data()

    test_and_visualize_gd()

    print("Training model with momentum = 0.0")
    svm = optimize_svm(train_data, train_targets, 1.0, GDOptimizer, 100, 500, 0.05, 0.0)
    reports(svm)

    print("Training model with momentum = 0.1")
    svm = optimize_svm(train_data, train_targets, 1.0, GDOptimizer, 100, 500, 0.05, 0.1)
    reports(svm)

import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

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


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    return 2/X.shape[0] * ( np.dot(np.dot(X.transpose(), X), w)
                            - np.dot(X.transpose(), y) )

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    
    true_gradient = lin_reg_gradient(X, y, w)
    
    batch_grad = 0
    for k in range(500):
        X_b, y_b = batch_sampler.get_batch(m=50)
        batch_grad += lin_reg_gradient(X_b, y_b, w) / 500

    print("Mean square error: ", np.mean((true_gradient - batch_grad)**2) )
    print("Cosine similarity: ", cosine_similarity(true_gradient, batch_grad))
    
    x_plot = []
    y_plot = []
    for m in range(1,401):
        w_j = []
        for k in range(500):
            X_b, y_b = batch_sampler.get_batch(m=m)
            w_j.append(lin_reg_gradient(X_b, y_b, w)[0])
        x_plot.append(np.log(m))
        y_plot.append(np.log(np.var(w_j)))
    
    plt.xlabel("log(m)")
    plt.ylabel("log(var(w[0]))")
    plt.plot(x_plot, y_plot)
    plt.show()


if __name__ == '__main__':
    main()

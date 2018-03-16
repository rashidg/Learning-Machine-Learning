'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    prep = [ [],[],[],[],[],[],[],[],[],[] ]
    for i in range(len(train_data)):
        x = train_data[i]
        y = train_labels[i]
        prep[int(y)].append(x)
    means = np.array([np.mean(klass, axis=0) for klass in prep])

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    def cov(bir, iki):
        a_mean = np.mean(bir)
        b_mean = np.mean(iki)
        sum = 0
        for i in range(0, len(bir)):
            sum += ((bir[i] - a_mean) * (iki[i] - b_mean))
        return sum/(len(bir)-1)

    covariances = np.zeros((10, 64, 64))
    prep = [ [],[],[],[],[],[],[],[],[],[] ]
    for i in range(len(train_data)):
        x = train_data[i]
        y = train_labels[i]
        prep[int(y)].append(x)

    for k in range(len(prep)):
        klass = np.array(prep[k])
        means = np.mean(klass, axis=0)
        for i in range(64):
            for j in range(64):
                covariances[k][i][j] = cov(klass[:,i], klass[:,j])
        covariances[k] += 0.01 * np.identity(64)

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    f, axarr = plt.subplots(ncols=10)
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        axarr[i].imshow(np.split(np.log(cov_diag), 8), cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    ret = []
    for i in range(len(digits)):
        res = []
        for k in range(10):
            digit = digits[i]
            mean = means[k]
            cov = covariances[k]
            d = 64
            a1 = -d/2  * np.log(2 * np.pi)
            a2 = -np.log(np.linalg.det(cov))/2
            a3 = (-1/2) * np.dot(np.dot(np.transpose(digit-mean), np.linalg.inv(cov)), (digit-mean))
            res.append(a1 + a2 + a3)

        ret.append(res)
    return np.array(ret)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class

    log p(x|y) + log p(y) - log p(x)
    p(y) = 1/10
    p(x) - marginal probability

    '''
    ret = []
    likelihoods = generative_likelihood(digits, means, covariances)
    for i in range(len(digits)):
        digit = digits[i]
        likelihood = likelihoods[i]
        divide_by = 0
        for k in range(10):
            divide_by += np.exp(likelihood[k])/10

        res = []
        for k in range(10):
            res.append(likelihood[k] + np.log(1/10) - np.log(divide_by))

        ret.append(res)
    return np.array(ret)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihoods = conditional_likelihood(digits, means, covariances)
    sum = 0
    for i in range(len(digits)):
        sum += cond_likelihoods[i][int(labels[i])]
    return sum / len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    res = []
    for i in range(len(digits)):
        res.append(cond_likelihood[i].argmax())
    return res

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    plot_cov_diagonal(covariances)

    #### Accuracy tests ####
    pred = classify_data(train_data, means, covariances)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == train_labels[i]:
            correct += 1
    print("train accuracy", correct / len(train_labels))

    pred = classify_data(test_data, means, covariances)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == test_labels[i]:
            correct += 1
    print("test accuracy", correct / len(test_labels))
    ########################

    print("avg train", avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("avg test", avg_conditional_likelihood(test_data, test_labels, means, covariances))


if __name__ == '__main__':
    main()

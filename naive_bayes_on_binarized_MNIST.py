'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    prep = [ [],[],[],[],[],[],[],[],[],[] ]
    for i in range(len(train_data)):
        x = train_data[i]
        y = train_labels[i]
        prep[int(y)].append(x)

    for i in range(10):
        klass = np.array(prep[i])
        eta[i] = klass.sum(axis=0) / len(klass)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    f, axarr = plt.subplots(ncols=10)
    for i in range(10):
        img_i = np.split(class_images[i], 8)
        axarr[i].imshow(img_i, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    generated_data = []
    for k in range(10):
        cur = []
        for j in range(64):
            cur.append(np.random.binomial(1, eta[k, j]))
        generated_data.append(cur)

    plot_images(np.array(generated_data))

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    '''
    ret = []
    for i in range(len(bin_digits)):
        res = []
        b = bin_digits[i]
        for k in range(10):
            ik = 0
            for j in range(64):
                ik += b[j] * np.log(eta[k][j]) + (1 - b[j]) * np.log(1-eta[k][j])
            res.append(ik)
        ret.append(res)
    return ret

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    ret = []
    likelihoods = generative_likelihood(bin_digits, eta)
    for i in range(len(bin_digits)):
        digit = bin_digits[i]
        likelihood = likelihoods[i]
        divide_by = 0
        for k in range(10):
            divide_by += np.exp(likelihood[k])/10

        res = []
        for k in range(10):
            res.append(likelihood[k] + np.log(1/10) - np.log(divide_by))

        ret.append(res)
    return np.array(ret)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihoods = conditional_likelihood(bin_digits, eta)
    sum = 0
    for i in range(len(bin_digits)):
        sum += cond_likelihoods[i][int(labels[i])]
    return sum / len(bin_digits)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    res = []
    for i in range(len(bin_digits)):
        res.append(cond_likelihood[i].argmax())
    return res

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # PREPROCESSING: add 2 extreme datapoints for each klass
    for k in range(10):
        train_data = np.concatenate((train_data, np.array([np.zeros(64), np.ones(64)])))
        train_labels = np.concatenate((train_labels, np.array([k, k])))

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)



    #### Accuracy tests ####
    pred = classify_data(train_data, eta)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == train_labels[i]:
            correct += 1
    print("train accuracy", correct / len(train_labels))

    pred = classify_data(test_data, eta)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == test_labels[i]:
            correct += 1
    print("test accuracy", correct / len(test_labels))
    ########################

    print("avg train", avg_conditional_likelihood(train_data, train_labels, eta))
    print("avg test", avg_conditional_likelihood(test_data, test_labels, eta))

    generate_new_data(eta)

if __name__ == '__main__':
    main()

'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())

        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        digit = None
        dists = self.l2_distance(test_point)
        labels = self.train_labels[dists.argsort()][:k]
        unique, counts = np.unique(labels, return_counts=True)
        digit = unique[counts.argmax()]

        return digit

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Return the optimal K

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    from sklearn.model_selection import KFold

    max_accuracy = 0
    best_k = -1
    for k in k_range:
        kfold = KFold(n_splits=10)
        accuracy = []
        for train_index, test_index in kfold.split(train_data):
            knn = KNearestNeighbor(train_data[train_index], train_labels[train_index])
            accuracy.append(classification_accuracy(knn, k, train_data[test_index], train_labels[test_index]))

        if np.mean(accuracy) > max_accuracy:
            max_accuracy = np.mean(accuracy)
            best_k = k
        print("  --> Average accuracy for K =", k, "is ", np.mean(accuracy))

    return best_k

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    hit = 0
    N = len(eval_data)
    for i in range(len(eval_data)):
        cur_data = eval_data[i]
        cur_label = eval_labels[i]
        predict = knn.query_knn(cur_data, k)
        if cur_label == predict:
            hit += 1

    return hit/N

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    print("K=1")
    print("training accuracy: ", classification_accuracy(knn, 1, train_data, train_labels))
    print("test accuracy: ", classification_accuracy(knn, 1, test_data, test_labels))

    print("K=15")
    print("training accuracy: ", classification_accuracy(knn, 15, train_data, train_labels))
    print("test accuracy: ", classification_accuracy(knn, 15, test_data, test_labels))

    print("best K is", cross_validation(train_data, train_labels))

if __name__ == '__main__':
    main()

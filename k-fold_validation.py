# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.misc import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses
 
 
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    # w* = (X'AX + ∂I)^-1 X'Ay
    # 
    # (X'AX + ∂I) w = X'Ay
    #
    # need to find A first
    # A[i,i] = a(i)
    # a(i) = exp(-||x-x(i)||**2 / 2t**2) /
    #       / ∑j exp(-||x-x(j)||**2 / 2t**2)
    # 
    # l2 gives ||x(i)-x(j)||**2
    #
    dists = l2(test_datum.transpose(), x_train)[0]
    denum = np.exp(logsumexp([ -d / (2 * tau**2) for d in dists]))
    A = np.zeros((x_train.shape[0], x_train.shape[0]))
    for i in range(x_train.shape[0]):
        A[i][i] = np.exp(-dists[i] / (2 * tau**2)) / denum
    
    a = np.add(np.dot(np.dot(x_train.transpose(), A), x_train),
               np.dot(lam, np.identity(x_train.shape[1])))
    b = np.dot(np.dot(x_train.transpose(), A), y_train)
    w = np.linalg.solve(a,b)

    return np.squeeze(np.dot(w, test_datum))


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    losses = np.zeros(taus.shape)
    chunk_size = x.shape[0] // k
    i = 0
    j = 0
    while i < x.shape[0]:
        j = min(i + chunk_size, x.shape[0])
        if x.shape[0]-10 < j:
            j = x.shape[0]
        x_test = x[idx[i:j]]
        y_test = y[idx[i:j]]
        x_train = np.concatenate([x[idx[0:i]], x[idx[j:x.shape[0]]]])
        y_train = np.concatenate([y[idx[0:i]], y[idx[j:y.shape[0]]]])
        loss = run_on_fold(x_test, y_test, x_train, y_train, taus)
        losses += loss / k
        i = j
        print("fold complete")
    return losses


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(losses)
    plt.show()
    print("min loss = {}".format(losses.min()))


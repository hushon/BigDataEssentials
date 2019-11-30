'''
Hyounguk Shon
17-Nov-2019

Usage: python svm.py [features.txt] [labels.txt]

Support Vector Machine with gradient descent fitting algorithm.
Training implements k-fold cross validation.

Example: http://www.di.kaist.ac.kr/~swhang/ee412/svm.zip
'''

import sys
import os
# import time
import numpy as np

def parse_feature(l):
    '''
    Arg:
        l (str)
    Return: 
        A list
    '''
    feature = l.split(',')
    feature = map(float, feature)
    feature = np.array(feature)
    return feature

def parse_label(l):
    '''
    Arg:
        l (str)
    Return: 
        A list
    '''
    label = [float(l)]
    label = np.array(label)
    return label

def split_dataset(x, n):
    '''
    split a list into n approximately-equal lengths of subsets.
    used to split dataset for K-fold cross-validation.
    Args:
        x: input list object
        n (int): number of chunks
    Returns:
        a list of n chunks, each with approximately equal size.
    '''
    k, m = divmod(len(x), n)
    return [x[i*k + min(i, m) : (i+1)*k + min(i+1, m)] for i in range(n)]

class SVM:
    '''
    define a SVM model.
    '''
    def __init__(self, feature_size, initializer):
        '''
        Args:
            feature_size (int): size of feature vector
            initializer : a function with shape as input parameter
        '''
        self.W = initializer((feature_size, 1))
        self.b = initializer((1, 1))

    def predict(self, x):
        '''
        perform inference from features.
        x shape: (N, nFeatures)
        Returns:
            class prediction; value is either +1 or -1
        '''
        W = self.W
        b = self.b
        return np.sign(np.matmul(x, W) + b)

    def fit(self, x, label, C=0.1, eta=0.2):
        '''
        train model by gradient descent.
        x shape: (N, nFeatures)
        label shape: (N, 1)
        Args:
        Returns:
            loss value
        '''
        assert isinstance(x, np.ndarray)

        # trick to regard b as a part of weight vector
        _W = np.concatenate([self.W, self.b], axis=0)
        _x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

        # calculate gradient vector
        det = (label * np.matmul(_x, _W)) < 1
        delta = np.sum(det * (- label * _x), axis=0, keepdims=True).transpose()
        grad = _W + C*delta

        # update weights
        _W = _W - eta * grad

        self.W = _W[:-1, :]
        self.b = _W[-1:, :]

        # calculate new loss
        loss = 0.5 * np.sum(self.W**2) + C*np.sum(np.clip(1 - label*np.matmul(x, self.W) + self.b, None, 0.0))
        return loss

def main():
    ''' parameters '''
    filepath_feature = sys.argv[1]
    filepath_label = sys.argv[2]
    k_fold = 10
    feature_size = 122 # feature vector size
    C = 0.01 # loss parameter
    eta = 0.01 # learning rate
    max_iter = 20

    ''' read and parse dataset '''
    with open(filepath_feature, 'r') as file:
        lines = file.readlines()
        features = map(parse_feature, lines)

    with open(filepath_label, 'r') as file:
        lines = file.readlines()
        labels = map(parse_label, lines)

    ''' transform dataset for k-fold cross validation '''
    features = split_dataset(features, k_fold)
    labels = split_dataset(labels, k_fold)
    features = np.array(features)
    labels = np.array(labels)

    ''' train model with k-fold cross validation '''
    acc_list = []
    k_list = list(range(k_fold))

    for _ in range(k_fold):
        # generate train and val set
        features_train = np.concatenate(features[k_list[1:]], axis=0)
        labels_train = np.concatenate(labels[k_list[1:]], axis=0)
        features_val = np.concatenate(features[k_list[:1]], axis=0)
        labels_val = np.concatenate(labels[k_list[:1]], axis=0)
        k_list = k_list[1:] + k_list[:1]

        # initialize SVM model
        initializer = lambda shape: np.random.normal(loc=0.0, scale=0.2, size=shape)
        model = SVM(feature_size, initializer)

        # train model
        for _ in range(max_iter):
            loss = model.fit(features_train, labels_train, C, eta)

        # calculate validation accuracy
        acc = np.sum(np.isclose(labels_val, model.predict(features_val))) / float(len(labels_val))
        acc_list.append(acc)
        # print '{0:.4f}'.format(acc)

    ''' print results '''
    print '{}'.format(np.mean(acc_list))
    print '{}'.format(C)
    print '{}'.format(eta)

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]), 'Cannot find file.'
    
    # starttime = time.time()
    main()
    # print 'Executed in: {}'.format(time.time()-starttime)
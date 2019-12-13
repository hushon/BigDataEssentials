'''
Hyounguk Shon
30-Nov-2019

Usage: python neuralnet.py [training.csv] [testing.csv]

Fully-connected Neural Network with gradient descent algorithm.

Example: http://www.di.kaist.ac.kr/~swhang/ee412/mnist_sample.zip
'''

import sys
import os
import time
import numpy as np

def parse(l):
    '''
    Arg:
        l (str)
    Return: 
        A list
    '''
    l = l.split(',')
    l = filter(bool, l)
    feature, label = list(map(float, l[:-1])), int(float(l[-1]))
    feature, label = np.array(feature), np.array(label)
    return feature, label

def encode_onehot(i, n):
    return np.eye(n)[i]

def sigmoid(x):
    x = np.array(x)
    return 1.0 / (1.0 + np.exp(-x))


class Fully_Connected_Network:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate

        '''Weight Initialization'''
        self.W1 = np.random.normal(loc=0.0, scale=0.2, size=(self.InputDim, self.HiddenDim))
        self.W2 = np.random.normal(loc=0.0, scale=0.2, size=(self.HiddenDim, self.OutputDim))

    def Forward(self, Input):
        '''Implement forward propagation'''
        ''' Input.shape is (N, InputDim) '''
        self.h1 = sigmoid(np.matmul(Input, self.W1))
        Output = sigmoid(np.matmul(self.h1, self.W2))
        return Output

    def Backward(self, Input, Label, Output):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''

        # dL/do
        grad = Output - Label
        
        # do/du
        grad *= Output*(1 - Output)

        # update value for self.W2
        W2_diff = self.learning_rate * np.matmul(self.h1.T, grad)

        grad = np.matmul(grad, self.W2.T)

        # update value for self.W1
        W1_diff = self.learning_rate * np.matmul(Input.T, grad)

        # update weights
        self.W2 -= W2_diff
        self.W1 -= W1_diff

    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output)

def main():
    ''' parameters '''
    filepath_train = sys.argv[1]
    filepath_test = sys.argv[2]
    feature_size = 784
    output_size = 10
    lr = 1.0
    decay_rate = 0.95
    max_epoch = 50

    ''' read and parse dataset '''
    with open(filepath_train, 'r') as file:
        trainset = file.read().splitlines()
        trainset = map(parse, trainset)

    with open(filepath_test, 'r') as file:
        testset = file.read().splitlines()  
        testset = map(parse, testset)

    train_feature = [x for x, y in trainset]
    train_label = [encode_onehot(y, output_size) for x, y in trainset]
    test_feature = [x for x, y in testset]
    test_label = [encode_onehot(y, output_size) for x, y in testset]

    '''Construct a fully-connected network'''
    Network = Fully_Connected_Network(lr)

    '''Train the network for the number of iterations'''
    # SGD with decaying lr
    indices = list(range(len(train_feature)))
    for _ in range(max_epoch):
        np.random.shuffle(indices)
        Network.learning_rate *= decay_rate
        for i in indices:
            Network.Train(
                train_feature[i][np.newaxis, ...], 
                train_label[i][np.newaxis, ...]
                )

    ''' evaluate training accuracy '''
    prediction = Network.Forward(train_feature)
    train_acc = np.mean(np.argmax(prediction, axis=1) == np.argmax(train_label, axis=1))

    ''' evaluate test accuracy '''
    prediction = Network.Forward(test_feature)
    test_acc = np.mean(np.argmax(prediction, axis=1) == np.argmax(test_label, axis=1))

    ''' print result '''
    print "{}".format(train_acc)
    print "{}".format(test_acc)
    print "{}".format(max_epoch)
    print "{}".format(lr)

if __name__ == '__main__':
    ''' sanity check '''
    assert os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2]), 'Cannot find file.'
    
    starttime = time.time()
    main()
    # print 'Executed in: {}x'.format(time.time()-starttime)
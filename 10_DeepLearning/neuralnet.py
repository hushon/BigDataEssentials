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
    feature, label = list(map(float, l[:-1])), int(l[-1])
    feature, label = np.array(feature), np.array(label)
    return feature, label

def encode_onehot(i, n):
    ''' encode label into length-n one-hot vector '''
    return np.eye(n)[i]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate

        '''Weight Initialization'''
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim)

    def Forward(self, Input):
        '''Implement forward propagation'''
        ''' Input.shape is (N, InputDim) '''
        h1 = np.matmul(Input, self.W1)
        h2 = np.matmul(h1, self.W2)
        Output = softmax(h2)
        return Output

    def Backward(self, Input, Label, Output):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''

    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output)

def main():
    ''' parameters '''
    filepath_train = sys.argv[1]
    filepath_test = sys.argv[2]
    feature_size = 122 # feature vector size
    lr = 1e-3
    max_iter = 20

    ''' read and parse dataset '''
    with open(filepath_train, 'r') as file:
        trainset = file.readlines()
        trainset = map(parse, trainset)

    with open(filepath_test, 'r') as file:
        testset = file.readlines()
        testset = map(parse, testset)

    '''Construct a fully-connected network'''
    Network = Fully_Connected_Layer(lr)

    '''Train the network for the number of iterations'''
    '''Implement function to measure the accuracy'''
    for i in range(max_iter):
        Network.Train(train_data, train_label)


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
import numpy as np

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
        return Output
    
    def Backward(self, Input, Label, Output):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''
    
    def Train(self, Input, Label):
        Output = self.Forward(Input)
        self.Backward(Input, Label, Output)        

'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

'''Train the network for the number of iterations'''
'''Implement function to measure the accuracy'''
for i in range(iteration):
    Network.Train(train_data, train_label)
    

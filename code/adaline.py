'''
NOTATION:
m = number of features
n = number of samples for training
'''

import numpy as np
import randomdata

class Adaline:
    '''
    Implementation of Adaline model. Neural Network Implementation
    
    Inputs:
        data: pd dataframe:
            all but one of the cols represent numerical features
            last column must be the target
    '''
    def __init__(self, data):
        '''
        Attributes:
            self.X: numericized matrix representing features
            self.y: series representing target
            self.w: vector of m+1 features, initialized randomly
                - n+1 because we want to include bias term
        '''
        # Need a feature that represents bias term
        # this will just be all 1 for everything in df
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]
        bias_x = np.full(X.iloc[:, 0].shape, 1)
        data.insert(0, 'bias', bias_x)

        # self.data is pd dataframe
        # self.X has shape (n, m + 1)
        # self.y has shape (n, 1)
        self.data = data
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]
        self.n = len(self.data)
        self.n_features = len(self.X.columns) # counts bias

        # need to randomly initialize initial weights.
        # self.w has shape (m, 1)
        rand = np.random.RandomState(42)
        self.w = rand.normal(scale=.01, size = (self.n_features, ))
    
    def __repr__(self):
        return str(self.w)
    
    def _net_inputs(self, to_test=None):
        '''
        get predicted value according to linear aggregation function.

        Input:
            self
        
        Output:
            vector of net_inputs.
                - takes shape (n, 1)

        net_input defined as linear combination of weight and feature
        values. 
        '''
        if to_test is None:
            return np.dot(self.X, self.w)
        else:
            return np.dot(to_test, self.w)
    
    def _threshold(self, to_test=None):
        '''
        Activation function for binary classification.
        Will be defined as such:
        1 if net_input > 0
        -1 if net_input <= 0

        Input:
            self
        
        Output:
            vector filled w/ either 1 or -1
                - takes shape (n, )
        '''
        if to_test is None:
            net_inputs = self._net_inputs()
            vec = np.vectorize(lambda x: 1 if x > 0 else -1)
            return vec(net_inputs)
        else:
            net_inputs = self._net_inputs(to_test)
            vec = np.vectorize(lambda x: 1 if x > 0 else -1)
            return vec(net_inputs)
    
    # tanh
    def _activation1(self):
        return np.tanh(self._net_inputs())
    
    # step function
    def _activation2(self):
        return self._threshold()
    
    def _sse(self):
        '''
        Is an example of a loss function. In this case, it will
        be the sum of squared errors, defined as:
        1/2 sum of (target - predicted)^2 for each sample.

        This is mainly a function of the weights. 

        Input:
            self
        
        Output:
            s (int): sum of squared errors
        '''
        predicted = self._activation1()
        summation = np.sum(np.square(self.y - predicted))
        return summation/2
        

    def _sse_grad(self):
        '''
        Get the gradient of the sse loss function with respect to 
        each weight. This will return a vector of m elements. 

        Input:
            self
        
        Output: 
            nabla_sse (numpy array):
                vector of slope of each w_j, (represents gradient of sse)
        '''
        predicted = self._activation1()
        inside_term = (self.y - predicted)
        return -np.dot(inside_term, self.X)
    
    def train(self, cost_func='sse', learning_rate=1e-3):
        # ensure no immediate convergence
        prev_w = np.array([100 for i in range(self.n_features)])
        while not np.allclose(prev_w, self.w, atol=1e-2):
            gradient = self._sse_grad()
            prev_w = self.w.copy()
            self.w += -learning_rate * gradient

        return self.w

    def predict(self, points):
        '''
        point should be an array of m features in correct order
        '''
        return self._threshold(points)
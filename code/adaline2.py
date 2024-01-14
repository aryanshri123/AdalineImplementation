'''
QUESTIONS:

Does initializing weight proportional to features make training faster?
e.g. weight feature so much larger than wingspan, should I initialize
weight's weight to be smaller compared to bias and wingspan?
'''

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

        self.data = data
        self.X = data.iloc[:, 0:-1]
        self.y = data.iloc[:, -1]

        # need to randomly initialize initial weights.
        rand = np.random.RandomState(42)
        self.w = rand.normal(scale=.01, size = self.X.shape[1])
    
    def __repr__(self):
        return str(self.w)
    
    def _net_inputs(self, to_test=None):
        '''
        get predicted value according to linear aggregation function.

        Input:
            self
        
        Output:
            vector of net_inputs.
                - takes shape (n, )

        net_input defined as linear combination of weight and feature
        values. 
        '''
        if to_test is None:
            return np.dot(self.X, self.w)
        else:
            return np.dot(to_test, self.w)

    def _threshold(self, to_test=None):
        '''
        Threshold function for binary classification.
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
    
    def _sse(self):
        '''
        Is an example of a loss function. In this case, it will
        be the sum of squared errors, defined as:
        sum of (target - predicted)^2 for each sample.

        This is mainly a function of the weights. 

        Input:
            self
        
        Output:
            s (int): sum of squared errors
        '''
        net_inputs = self._net_inputs()
        diff_squared = np.square(self.y - net_inputs)
        return np.sum(diff_squared)
    
    def _mse(self):
        '''
        Is an example of a loss function. In this case, it will
        be the mean of squared errors. This is identical to sse,
        but we divide sum by number of samples. 

        Again, we are mainly interested in the weights

        Input:
            self
        
        Output:
            m (float): mean of squared errors
        '''
        m = self._sse()
        return m/(self.X.shape[0])

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
        # partial derivative of sse with respect to each w_j:
        # - (sum of (2 * (y - net_input) * x_j)
        # y is a vector of n elements
        # net_input is a vector of n elements
        # x_j is a vector of n elements
        net_inputs = self._net_inputs()
        diff = self.y - net_inputs
        multiplied_diff = 2 * diff
        
        nabla_sse = np.array([])
        for i in range(len(self.w)):
            curr_feature = self.X.iloc[:, i]
            result = -np.dot(multiplied_diff, curr_feature)
            nabla_sse = np.append(nabla_sse, result)

        return nabla_sse

    def _mse_grad(self):
        '''
        Get gradient of mse loss function with respect to each weight
        This will return a vector of m elements.

        Input:
            self
        
        Output:
            nabla_mse (numpy array):
                vector of slope of each w_j (represents gradient for mse)
        '''
        nabla_sse = self._sse_grad()
        return nabla_sse/(self.X.shape[0])
    
    def train(self, cost_func='mse', learning_rate=1e-3, n_iter=5000):
        # if cost_func == 'mse':
        #     gradient = self._mse_grad
        # else:
        #     gradient = self._sse_grad
        for i in range(n_iter):
            nabla = self._mse_grad()
            print(nabla)
            #print(i, learning_rate * nabla, self.w)
            self.w += (-learning_rate * nabla)
        
        # return self.w
    
    def predict(self, points):
        '''
        point should be an array of m features in correct order
        '''
        return self._threshold(points)
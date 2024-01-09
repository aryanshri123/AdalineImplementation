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
    
    def _net_inputs(self):
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
        return np.dot(self.X, self.w)

    def _threshold(self):
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
        net_inputs = self._net_inputs()
        vec = np.vectorize(lambda x: 1 if x > 0 else -1,)
        return vec(net_inputs)
    
    def _loss_func(self):
        '''
        Is the loss function for the model. In this case, it will
        be the sum of squared errors, defined as:
        sum of (target - predicted)^2 for each sample.

        This is mainly a function of the weights

        Input:
            self
        
        Output:
            err (int): sum of squared errors
        '''
        net_inputs = self._net_inputs()
        diff_squared = np.square(self.y - net_inputs)
        return np.sum(diff_squared)


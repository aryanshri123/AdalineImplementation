import numpy as np

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
            self.data (DataFrame): original data provided with bias term included
            self.X (DataFrame): numericized matrix representing features
            self.y (Series): series representing target
            self.w (numpy vector): vector of features, initialized randomly
            self.n (int): number of samples
            self.n_features (int): number of features, including bias term artifically added
        '''
        # Need a feature that represents bias term
        # this will just be all 1 for everything in df
        X = data.iloc[:, 0:-1]
        y = data.iloc[:, -1]
        bias_x = np.full(X.iloc[:, 0].shape, 1)
        data.insert(0, 'bias', bias_x)

        # self.data is pd dataframe
        # self.X has shape (n, m + 1)
        #     n is n
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
            to_test (DataFrame): dataframe of similar structure to inputted dataframe,
                                 without targets
        
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
            to_test (DataFrame): dataframe of similar structure to inputted dataframe,
                                 without targets
        
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
    
    def _sse(self, activation='tanh'):
        '''
        Is an example of a loss function. In this case, it will
        be the sum of squared errors, defined as:
        1/2 sum of (target - predicted)^2 for each sample.

        This is mainly a function of the weights. 

        Input:
            self
            activation (str): either tanh or step
        
        Output:
            (int): sum of squared errors / 2
                   Taking over 2 to simplify gradient calculation
        '''
        if activation == 'tanh':
            predicted = self._activation1()
        elif activation == 'step':
            predicted = self._activation2()
        summation = np.sum(np.square(self.y - predicted))
        return summation/2
        

    def _sse_grad(self, activation='tanh'):
        '''
        Get the gradient of the sse loss function with respect to 
        each weight. This will return a vector of m elements. 

        Input:
            self
            activation (str): either tanh or step activation
        
        Output: 
            (numpy array): vector of slope of each w_j, (represents gradient of sse)
        '''
        if activation == 'tanh':
            predicted = self._activation1()
        elif activation == 'step':
            predicted = self._activation2()
        inside_term = (self.y - predicted)
        return -np.dot(inside_term, self.X)
    
    def train(self, learning_rate=1e-3, max_iter=5000, activation='tanh'):
        '''
        Train the model by updating weights by gradient descent.

        Inputs:
            self
            learning_rate (float): initialize to small number, ensures we don't
                                   overshoot
            max_iter (int): Maximum iterations before we "give" up on finding
                            weights to converge to. 

        Output:
            self.w (numpy array): vector of weights to use.
        '''
        # ensure no immediate convergence
        prev_w = np.array([100 for i in range(self.n_features)])
        i = 0
        while not np.allclose(prev_w, self.w, atol=1e-2) and i < max_iter:
            gradient = self._sse_grad(activation)
            prev_w = self.w.copy()
            self.w += -learning_rate * gradient
            i += 1

        return self.w

    def predict(self, to_test):
        '''
        point should be an array of m features in correct order

        Input:
            self
            to_test (DataFrame): samples to predict on in similar format to inputted DataFrame
                                 Should not include bias or target columns.
        '''
        # Introduce bias term
        bias_x = np.full(to_test.iloc[:, 0].shape, 1)
        to_test.insert(0, 'bias', bias_x)

        return self._threshold(to_test)

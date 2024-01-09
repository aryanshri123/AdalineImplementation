import numpy as np
import randomdata

n_samples = 100

# albatross
alb_mean_weight = 9000
alb_var_weight = 800
alb_mean_wingspan = 300
alb_var_wingspan = 20
alb_target = 1

# owls
owl_mean_weight = 1000
owl_var_weight = 200
owl_mean_wingspan = 100
owl_var_wingspan = 15
owl_target = -1

alb_data = randomdata.species_gen(alb_mean_weight, alb_var_weight, alb_mean_wingspan, alb_var_wingspan, n_samples, alb_target, 42)
owl_data = randomdata.species_gen(owl_mean_weight, owl_var_weight, owl_mean_wingspan, owl_var_wingspan, n_samples, owl_target, 42)


def initialize_weights(X):
    '''
    Initialize x_0 to be 1 because we want a bias term (to be represented by w[0])


    '''
    #return X.shape[0]
    bias_x = np.full(X.shape[0], 1)
    #X = np.add(bias, X)
    X = np.c_[bias_x, X]

    rand = np.random.RandomState(42)
    w = rand.normal(scale=.01, size=X.shape[1])
    return X, w

def net_input(X, w):
    '''
    Gets prediction y^ (referred to as net input)

    This is just a dot product of the thing

    Returns a vector of net_input
    '''
    return np.dot(X, w)

def loss_function(data, w):
    '''
    SSE (sum of squared errors)
    '''
    X = data[0]
    y = data[1]
    # X.shape[1] is num samples
    return (np.sum(np.square((y - net_input(X,w)))))/(X.shape[0])

def gradient(learning_rate, data, w):
    

# def train(learning_rate, n_iter=100):
#     for 

alb_weights = initialize_weights(alb_data[0])
owl_weights = initialize_weights(owl_data[0])
print(gradient(.05, (alb_weights[0], alb_data[1]), alb_weights[1]))



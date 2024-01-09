import numpy as np

def species_gen(mean_f1, var_f1, mean_f2, var_f2, n_samples, target, seed):
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mean_f1, var_f1, n_samples)
    f2 = rand.normal(mean_f2, var_f2, n_samples)
    X = np.array([f1, f2]).transpose()
    y = np.full(n_samples, target)
    return X, y

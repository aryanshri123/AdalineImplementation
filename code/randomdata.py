import numpy as np
import pandas as pd

def species_gen(mean_f1, var_f1, mean_f2, var_f2, n_samples, target, seed=42):
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mean_f1, var_f1, n_samples)
    f2 = rand.normal(mean_f2, var_f2, n_samples)
    X = np.array([f1, f2]).transpose()
    y = np.full(n_samples, target)
    data = np.c_[X, y]
    return pd.DataFrame(data)

'''
     bias       weight    wingspan
0       1  7829.188041  307.706348
1       1  8550.169977  306.030947
2       1  8518.634710  268.986731
3       1  1066.252686   82.130448
4       1   879.872262  101.494770
..    ...          ...         ...
195     1  1315.842563  128.292789
196     1  7620.065734  296.152781
197     1  8438.357525  304.281875
198     1   602.486217  140.802537
199     1  1129.537708   94.859282
'''
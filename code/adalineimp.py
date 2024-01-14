import numpy as np
import pandas as pd
import randomdata
from adaline2 import Adaline

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
full_data = np.r_[alb_data, owl_data]

df = pd.DataFrame(full_data, columns=['weight', 'wingspan', 'target'])
df = df.sample(frac = 1, random_state = 42, ignore_index = True)


alb_test = randomdata.species_gen(alb_mean_weight, alb_var_weight, alb_mean_wingspan, alb_var_wingspan, n_samples, alb_target, 142)
owl_test = randomdata.species_gen(owl_mean_weight, owl_var_weight, owl_mean_wingspan, owl_var_wingspan, n_samples, owl_target, 142)

full_test = np.r_[alb_test, owl_test]
test_df = pd.DataFrame(full_test, columns=['weight', 'wingspan', 'target'])
test_df = test_df.sample(frac = 1, random_state = 42, ignore_index = True)
test_df = test_df.drop('target',axis=1)
test_df['bias'] = np.full((len(test_df),), 1)
test_df['target'] = df['weight'].apply(lambda x: 1 if x > 5000 else -1)
cols = list(test_df.columns)
cols = ['bias', 'weight', 'wingspan', 'target']
test_df = test_df[cols]

model = Adaline(df)

def squared_err(row):
    z = np.dot(row[:-1], model.w)
    targ = row['target']
    return np.square(targ - z)


df['net_input'] = model._net_inputs(df.iloc[:, 0:-1])
print(df['target'] - df['net_input'])



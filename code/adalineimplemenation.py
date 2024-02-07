import numpy as np
import pandas as pd
import randomdata
from adaline import Adaline


# create dataset
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

alb_data_train = randomdata.species_gen(alb_mean_weight, alb_var_weight, \
                                  alb_mean_wingspan, alb_var_wingspan, \
                                  n_samples, alb_target, 42)
owl_data_train = randomdata.species_gen(owl_mean_weight, owl_var_weight, \
                                  owl_mean_wingspan, owl_var_wingspan, \
                                  n_samples, owl_target, 42)                            

df = pd.concat([alb_data_train, owl_data_train], ignore_index=True)
df = df.sample(frac=1.0, random_state=42)
df = df.reset_index()
df = df[df.columns.difference(['index'])]


model = Adaline(df)
weights = model.train(activation='tanh')
print(f'Model Weights: \n \
        bias: {weights[0]} \n \
        weight: {weights[1]} \n \
        wingspan: {weights[2]}')

alb_data_test = randomdata.species_gen(alb_mean_weight, alb_var_weight, \
                                  alb_mean_wingspan, alb_var_wingspan, \
                                  50, alb_target, 10)
owl_data_test = randomdata.species_gen(owl_mean_weight, owl_var_weight, \
                                  owl_mean_wingspan, owl_var_wingspan, \
                                  50, owl_target, 10)    

df_t = pd.concat([alb_data_test, owl_data_test], ignore_index=True)
df_t = df_t.sample(frac=1.0, random_state=42)
df_t = df_t.reset_index()
df_t = df_t[df_t.columns.difference(['index'])]

df_to_test = df_t.iloc[:, :-1]


predictions = pd.Series(model.predict(df_to_test))
df_to_test.insert(len(df_to_test.columns), 'predictions', predictions)

accuracy_series = (df_t.loc[:, 2] == df_to_test['predictions'])
print(f'Test Accuracy: {accuracy_series.value_counts().iloc[0]/len(accuracy_series)*100}%')
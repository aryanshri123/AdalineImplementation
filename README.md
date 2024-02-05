# Overview

This repository contains 3 python files: `adaline.py`, `adalineimplementation.py`, `randomdata.py`. `adaline.py` contains the actual adaline model. `randomdata.py` is a random data generator that is used in `adalineimplementation.py` to provide a specific example of how one can implement the ADALINE model found in `adaline.py`. 

# Notes

I implemented a tanh activation function because it better captures non-linearity in data as opposed to a linear activation function. The use of the hyperbolic tangent function also caused my weights to converge much faster as compared to a linear activation funciton. 

I also included a step activation function as I found that that also worked when testing with the specifications of `adalineimplementation.py`, although it converged much slower as compared to the use of the hyperbolic tangent function. 

# Next Steps

Test on real data with many more features.

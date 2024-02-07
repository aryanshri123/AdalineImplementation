# Overview

This repository contains 3 python files in the code folder: `adaline.py`, `adalineimplementation.py`, `randomdata.py`. `adaline.py` contains the actual adaline model. `randomdata.py` is a random data generator that is used in `adalineimplementation.py` to provide a specific example of how one can implement the ADALINE model found in `adaline.py`. 

# Notes

I implemented a tanh activation function because it better captures non-linearity in data as opposed to an identity activation function. The use of the hyperbolic tangent function also caused my weights to converge much faster as compared to an identity activation function because of its boundedness.
<img width="596" alt="tanh" src="https://github.com/aryanshri123/AdalineImplementation/assets/153876046/d97a151e-f207-4517-8936-9bf71640bee9">

I also included a step activation function as I found that that also worked when testing with the specifications of `adalineimplementation.py`, although it converged much slower as compared to the use of the hyperbolic tangent function. 
<img width="608" alt="step" src="https://github.com/aryanshri123/AdalineImplementation/assets/153876046/48766f0a-0923-4610-9ce1-e323d374f04f">

# Next Steps

Test on real data with many more features.

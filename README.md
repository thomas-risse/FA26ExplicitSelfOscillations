# Code for the paper "NON-ITERATIVE ENERGY BALANCED SCHEME FOR A CATEGORY OF SELF-OSCILLATING INSTRUMENTS" submitted to the Forum Acusticum 2026

The main goal of the paper is to propose a non-iterative method for the simulation of self-oscillating musical instruments including localized nonlinear dissipation laws (usually at the exciter position) and general conservative nonlinearities arising from non-quadratic potential energy.

This repo contains example code for the three simplified cases presented in the paper:
1. A bowed string. Here, the string model is considered linear for simplicity. A modal spatial discretization is used. This results in the need to solve a linear system at each time-step which is a rank 1 perturbation to a constant diagonal matrix, that can be solved using the Sherman-Morrison formula. A nonlinearity could be added to the string model itself, in which case a linear system with a rank 2 perturbation of a constant diagonal matrix would have to be solved at each time-step, using the Woodbury formula (need to compute the inverse of a $2\times 2$ matrix).
2. A single reed instrument. The exciter here is a scalar system such that solving the linear system is trivial. However, this case demonstrates the coupling with a resonator representing the bore of the instrument.
3. Voice, using the body-cover model of the vocal folds.

## Structure 
The repo contain two main folder `/python` containing the python code used to generate figures from the C++ code in `src`.
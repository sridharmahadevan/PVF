This is MATLAB code for proto-value functions (PVFs), a way to construct bases for reinforcement learning that adapts to nonlinear state space geometry. 

A detailed overview of PVFs is given in the Journal of Machine Learning Research (JMLR) paper, downloadable from: 

http://jmlr.csail.mit.edu/papers/volume8/mahadevan07a/mahadevan07a.pdf

For a sample demonstration, execute the following code: 

pendulum_rpi_generic_experiment(50,20,8,'pendulum_basis_eigen');


This MATLAB package requires the Parallel Distributed Toolbox, and also uses nVidia GPUs for faster processing. 


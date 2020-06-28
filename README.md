# Bug_Shooting

We call for help to shoot a bug in DistributedDataParallel. 
This repo is used to reproduce the problem, i.e., `RuntimeError: Expected to have finished reduction in the prior 
iteration before starting a new one.` By default, we use 2 GPUs to run the DistributedDataParallel training.
Any help is highly appreciated. Thanks.

# Environment
- pytorch 1.4.0
- python 3.7

# Objective
We would like to share some data across different processes, but it triggers the runtime error mentioned above.
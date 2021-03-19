# On the Extrapolation of Graph Neural Networks and Beyond
**Group members (Rolling a Dice)**: Yihe Deng, Yu Yang, Qing Li, Shuwen Qiu.

This repository hosts the code, the report, and the presentation slides for our CS-249-GNN final project in 2021 Winter.

## Prerequisites
* Ubuntu 20.04
* Python 3.6
* NVIDIA TITAN RTX + CUDA 10.0
* PyTorch 1.4.0
* [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)


## Reproducing Experiments
### n-body
The experiments on n-body are hosted in the folder <a href = "/n_body">n_body</a>
To generate data, we can run the following code:
```
cd n_body
python physics.py 
```
Please refer to the python file <a href = "/n_body/physics.py">physics.py</a> for setting various parameters, such as the the the number of stars, time steps, distance range, masses of stars.

To reproduce the experiment results, we can simply run the following code:
```
cd n_body
python main.py 
```
Please refer to the python file <a href = "/n_body/main.py">main.py</a> for setting various hyperparameters in the model, such as the the learning rate, loss functions.


### HINT
The experiments on HINT are hosted in the folder <a href = "/hint">hint</a>
To reproduce the experiment results, we can simply run the following code:
```
cd hint
python train.py --curriculum
```
Please refer to the python file <a href = "/hint/train.py">train.py</a> for setting various hyperparameters in the model, such as the number of layers and the learning rate.

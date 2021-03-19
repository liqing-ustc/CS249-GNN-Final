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
### max degree and shortest path
The experiments on the max degree task are hosted in the folder <a href = "/graph_algos">graph_algos</a>.

To generate data, run the following code:
- max degree
```
cd graph_algos
python maxdeg_generation.py 
```
- shortest path
```
cd graph_algos
python shortest_generation.py 
```
Please refer to the python file <a href = "/graph_algos/maxdeg_generation.py">maxdeg_generation.py</a> and <a href = "/graph_algos/shortest_generation.py">shortest_generation.py</a> for setting various parameters.

To reproduce the experiment results, run the following code:
- max degree
```
cd graph_algos
python main.py --train maxdeg --graph_pooling_type READOUT --neighbor_pooling_type AGGREGATE --subtype 8 --n_iter 1 --mlp_layer 2 --fc_output_layer 1 --loss_fn reg --epochs 300 
```
- shortest path
```
cd graph_algos
python main.py --train shortest --graph_pooling_type READOUT --neighbor_pooling_type AGGREGATE --subtype 14 --n_iter 3 --mlp_layer 2 --fc_output_layer 1 --loss_fn reg --epochs 250 
```
For both tasks, READOUT and AGGREGATE should be one of the following: max, sum, mean, min.

Please refer to the python file <a href = "/graph_algos/main.py">main.py</a> for setting various hyperparameters in the model.


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

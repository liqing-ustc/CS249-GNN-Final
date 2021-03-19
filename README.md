# On the Extrapolation of Graph Neural Networks and Beyond
**Group members (Rolling a Dice)**: Yihe Deng, Yu Yang, Qing Li, Shuwen Qiu.

This repository hosts the code, the [report](CS249_Project_Report.pdf), and the [presentation slides]() for our CS-249-GNN final project in 2021 Winter.

## Prerequisites
* Ubuntu 20.04
* Python 3.6
* NVIDIA TITAN RTX + CUDA 10.0
* PyTorch 1.4.0
* [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)


## Reproducing Experiments

### HINT
The experiments on HINT are hosted in the folder <a href = "/hint">hint</a>
To reproduce the experiment results, we can simply run the following code:
```
cd hint
python train.py --curriculum
```
Please refer to the python file <a href = "/hint/train.py">train.py</a> for setting various hyperparameters in the model, such as the number of layers and the learning rate.

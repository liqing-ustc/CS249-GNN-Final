import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def calc_output_size(args):
    if args.subtype == 8: #max_degree
        answer_size = 1
    elif args.subtype == 14: #length of the shortest path
        answer_size = 1
    elif args.subtype == 15: #color of the predecessor
        answer_size = args.num_colors
    return answer_size

def median_absolute_percentage_error_compute_fn(y_pred, y):
    e = torch.abs(y.view_as(y_pred) - y_pred) / torch.abs(y.view_as(y_pred))
    return 100.0 * torch.mean(e)

class S2VGraph(object):
    def __init__(self, label, node_features, neighbors, g=None):
        '''
            label: graph label
            neighbors: list of neighbors (without self-loop)
            node_features: a torch float tensor
            g: original networkX graph
        '''
        self.label = label
        self.neighbors = neighbors
        self.node_features = node_features
        self.g = g
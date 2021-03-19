import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from MLPs import *
from util import *
import copy

class RelationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
        return self.model(x)


class ObjectModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ObjectModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        '''
        Args:
            x: [n_objects, input_size]
        Returns:
            [n_objects, output_size]

        Note: output_size = number of states we want to predict
        '''
        return self.model(x)

class InteractionNetwork(BasicModel):
    def __init__(self, args, x_dim=0):
        super(InteractionNetwork, self).__init__(args, 'InteractionNetwork')
        
        self.device = args.device
        self.bs = args.batch_size
        self.n_objects = args.n_objects
        self.n_relations = self.n_objects * (self.n_objects - 1)
        self.obj_dim = args.node_feature_size

        self.rel_dim = args.edge_feature_size
        answer_size = args.answer_size
        self.eff_dim, hidden_rel_dim, hidden_obj_dim = args.hidden_dim, args.hidden_dim, args.hidden_dim
        self.rm = RelationModel(self.obj_dim * 2 + self.rel_dim, hidden_rel_dim, self.eff_dim)
        self.om = ObjectModel(self.obj_dim + self.eff_dim + x_dim, hidden_obj_dim, answer_size)  # x, y
        
        receiver_r = np.zeros((self.n_objects, self.n_relations), dtype=float)
        sender_r = np.zeros((self.n_objects, self.n_relations), dtype=float)

        count = 0   # used as idx of relations
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if i != j:
                    receiver_r[i, count] = 1.0
                    sender_r[j, count] = 1.0
                    count += 1

        self.rs = Variable(torch.FloatTensor(sender_r)).to(self.device)
        self.rr = Variable(torch.FloatTensor(receiver_r)).to(self.device)
        #self.ra = Variable(torch.FloatTensor(r_info)).to(self.device)
        
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.decay)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr, weight_decay=args.decay)

    def norm(self, a, b):
        return torch.norm(a - b, 2)

    def m(self, obj, bs):
        """
        The marshalling function;
        computes the matrix products ORr and ORs and concatenates them with Ra

        :param obj: object states
        :param rr: receiver relations
        :param rs: sender relations
        :param ra: relation info
        :return:
        """
        r_info = np.zeros((bs, self.rel_dim, self.n_relations))
        ra = Variable(torch.FloatTensor(r_info)).to(self.device)
        for b in range(bs):
            count = 0
            for i in range(self.n_objects):
                for j in range(self.n_objects):
                    if i != j:
                        s_mass = obj[b, j, 0]
                        s_pos = obj[b, j, 1:3]
                        r_pos = obj[b, i, 1:3]
                        ra[b, :, count] = s_mass* (s_pos - r_pos)/ self.norm(s_pos, r_pos)**3
                        count += 1

        obj_t = torch.transpose(obj, 1, 2).reshape(-1, self.n_objects) # (bs * obj_dim, n_objects)
        orr = obj_t.mm(self.rr).reshape((bs, self.obj_dim, -1))   # (bs, obj_dim, n_relations)
        ors = obj_t.mm(self.rs).reshape((bs, self.obj_dim, -1))    # (bs, obj_dim, n_relations)

        return torch.cat([orr, ors, ra], dim = 1)   # (bs, obj_dim*2+rel_dim, n_relations)

    def forward(self, obj, bs, x=None):
        """
        objects, sender_relations, receiver_relations, relation_info
        :param obj: (bs, n_objects, obj_dim)
        :param rr: (n_objects, n_relations)
        :param rs: (n_objects, n_relations)
        :param x: external forces, default to None
        :return:
        """
        # marshalling function
        b = self.m(obj, bs)   # shape of b = (bs, obj_dim*2+rel_dim, n_relations)
        # relation module
        e = self.rm(torch.transpose(b, 1, 2))   # shape of e = (bs, n_relations, eff_dim)
        e = torch.transpose(e, 1, 2).reshape(-1, self.n_relations)   # shape of e = (bs * eff_dim, n_relations)
        # effect aggregator
        if x is None:
            # shape of a = (bs, obj_dim+eff_dim, n_objects)
            a = torch.cat([torch.transpose(obj, 1, 2), e.mm(self.rr.t()).reshape((bs, self.eff_dim, -1))], dim = 1)   
        else:
            # [unchanged] shape of a = (obj_dim+ext_dim+eff_dim, n_objects) 
            a = torch.cat([torch.transpose(obj, 1, 2), x, e.mm(self.rr.t())])   
        # object module
        p = self.om(torch.transpose(a, 1, 2))   # shape of p = (bs, n_objects, answer_size)

        return p
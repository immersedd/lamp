import torch
import torch.nn as nn
from .util import prepare_trees
import numpy as np
from mlp import *


class TCNNDS(nn.Module):
    def __init__(self, in_channels,phi_hidden_dims= (128,),phi_output_dim=32,):
        super(TCNNDS, self).__init__()
        self.__in_channels = in_channels
        self.__cuda = True

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling()
        )

        self.deep_set_aggregator = DeepSetAggregator(
            input_dim=64,
            phi_hidden_dims=list(phi_hidden_dims),
            phi_output_dim=phi_output_dim,
            rho_hidden_dims=[64],
            output_dim=64,
            aggr='sum'
        )

        input_dim = 64
        hidden_dims = [32]
        output_dim = 2

        self.mlp = MLP(input_dim, hidden_dims, output_dim)

    def in_channels(self):
        return self.__in_channels

    def forward(self, *inputs):
        embeddings = []
        for x in inputs:
            trees = prepare_trees(x, features, left_child, right_child, cuda=self.__cuda)
            plan_emb = self.tree_conv(trees)  # shape: [B, 64]
            embeddings.append(plan_emb)

        stacked_emb = torch.stack(embeddings, dim=1)
        combined_emb = self.deep_set_aggregator(stacked_emb)

        output = self.mlp(combined_emb)
        mean, log_var = output[:, 0], output[:, 1]
        return mean, torch.exp(log_var)

    def cuda(self):
        self.__cuda = True
        return super().cuda()


class DeepSetAggregator(nn.Module):

    def __init__(self, input_dim, phi_hidden_dims, phi_output_dim, rho_hidden_dims, output_dim, aggr='sum'):
        super(DeepSetAggregator, self).__init__()
        self.phi = MLP(input_dim, phi_hidden_dims, phi_output_dim)
        self.rho = MLP(phi_output_dim, rho_hidden_dims, output_dim)
        assert aggr in ['mean', 'sum']
        self.aggr = aggr

    def forward(self, embeddings):
        B, n, d = embeddings.shape
        phi_out = self.phi(embeddings.view(-1, d))
        phi_out = phi_out.view(B, n, -1)
        if self.aggr == 'mean':
            aggregated = phi_out.mean(dim=1)
        else:
            aggregated = phi_out.sum(dim=1)
        output = self.rho(aggregated)
        return output

def left_child(x):
    if len(x) != 3:
        return None
    return x[1]

def right_child(x):
    if len(x) != 3:
        return None
    return x[2]

def features(x):
    return x[0]


from collections import OrderedDict
import numpy as np
import random as rd
import scipy.sparse as sp
import torch
import copy
from texttable import Texttable
from collections import defaultdict
from scipy.io import loadmat
import pickle

def args_print(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

def randint():
    return np.random.randint((1<<31) - 1)

def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'paper':
            attr = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, attr

def feature_reddit(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']), dtype=np.float)
        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'def':
            attr = feature[_type]
    return feature, times, indxs, attr

def pos_neg_split(nodes, labels):
    pos_nodes = []
    nodes = nodes.tolist()
    neg_nodes = copy.deepcopy(nodes)
    aux_nodes = copy.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])
    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    aux_nodes = copy.deepcopy(neg_nodes)
    aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes) * scale))
    batch_nodes = pos_nodes + aux_nodes
    return batch_nodes



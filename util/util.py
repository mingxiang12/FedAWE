import numpy as np
import torch
from config import dirichlet_alpha, fluctuate
from dataset.dataset import clip
from torch.utils.data import Dataset
from copy import deepcopy


class ClientSampler:
    def __init__(self, participation_prob):
        self.participation_prob = participation_prob

    def sample(self):
        return np.random.binomial(1, self.participation_prob) == 1
        
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def accumulation(base, delta, number, device, lr_global):
    return torch.div(base, torch.tensor(number).to(device)).view(-1) + lr_global * clip(torch.div(delta, torch.tensor(number).to(device)).view(-1), 0.5)

def evaluate(w_global, device, w_local = None):
    if w_local == None:
        return w_global
    else:
        return sum(w_local).to(device) / len(w_local)

def partition(dataset, num_clients):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    if isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    min_label = min(labels)
    max_label = max(labels)
    num_labels = max_label - min_label + 1

    label_distributions_each_node = np.random.dirichlet(dirichlet_alpha * np.ones(num_labels), num_clients)
    sum_prob_per_label = np.sum(label_distributions_each_node, axis=0)

    indices_per_label = []
    for i in range(min_label, max_label + 1):
        indices_per_label.append([j for j in range(len(labels)) if labels[j] == i])

    start_index_per_label = np.zeros(num_labels, dtype='int64')
    for n in range(num_clients):
        for i in range(num_labels):
            end_index = int(np.round(len(indices_per_label[i]) * np.sum(label_distributions_each_node[:n+1, i]) / sum_prob_per_label[i]))
            dict_users[n] = np.concatenate((dict_users[n], np.array(indices_per_label[i][start_index_per_label[i] : end_index], dtype='int64')), axis=0)
            start_index_per_label[i] = end_index

    actual_label_distributions_each_node = [np.array([len([j for j in labels[dict_users[n]] if j == i]) for i in range(min_label, max_label + 1)], dtype='int64')
                                            / len(dict_users[n]) for n in range(num_clients)]

    return dict_users, actual_label_distributions_each_node, num_labels


def data_participation_each_node(data_train, num_clients):
    dict_users, actual_label_distributions_each_node, num_labels = partition(data_train, num_clients)
    
    uniform_first_half = np.random.uniform(0,1,actual_label_distributions_each_node[0].shape[0] // 2).astype(float) 
    uniform_second_half = np.random.uniform(0,0.5,actual_label_distributions_each_node[0].shape[0] // 2).astype(float) 
    uniform_data = list(uniform_first_half)
    uniform_data.extend(list(uniform_second_half))
    
    p_each_client = np.array([np.sum(np.multiply(actual_label_distributions_each_node[n], uniform_data)) for n in range(num_clients)])

    return dict_users, p_each_client


def data_participation_each_node_per_round(participation_prob_each_node_init, r):
    if fluctuate == 1:
        participation_prob_each_node = ( .3 * np.sin(2*np.pi/20 * r)   + .7) * participation_prob_each_node_init

    elif fluctuate == 2:
        participation_prob_each_node = ( .2 * np.sin(2*np.pi/20 * r)   + .8) * participation_prob_each_node_init
    
    elif fluctuate == 3:    
        participation_prob_each_node = ( .1 * np.sin(2*np.pi/20 * r)   + .9) * participation_prob_each_node_init
    
    elif fluctuate == 4:
        participation_prob_each_node = np.zeros_like(participation_prob_each_node_init)
        values = ( .3 * np.sin(2*np.pi/20 * r )   + .7) * participation_prob_each_node_init
        
        for id, v in enumerate(values):
            if v < 0.1:
                pass
            else:
                participation_prob_each_node[id] = v
    
    elif fluctuate == 5:
        if np.mod((r//10),2) == 1:
            participation_prob_each_node = .4 * participation_prob_each_node_init
        else:
            participation_prob_each_node = participation_prob_each_node_init
    
    elif fluctuate == 6:
        participation_prob_each_node = participation_prob_each_node_init

    else:
        raise NotImplementedError
        
    participation_prob_each_node = np.maximum(np.minimum(1, participation_prob_each_node),0)
    
    return participation_prob_each_node


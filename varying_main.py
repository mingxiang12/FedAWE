import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from config import *
from dataset.dataset import *
from statistic.collect_stat import stats_collector
from util.util import data_participation_each_node, data_participation_each_node_per_round, DatasetSplit, ClientSampler
import numpy as np
import random
import copy
from model.model import Model

def varying_dynamics():
    stat = stats_collector(prefix=prefix)

    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)  
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed)  
        torch.backends.cudnn.deterministic = True  

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=256, num_workers=0)
        dict_users, p_each_client = data_participation_each_node(data_train, num_clients)        

        step_size_local = lr_local_init

        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True)

        train_loader_list = []
        dataiter_list = []
        for n in range(num_clients):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = next(dataiter_list[n])
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = next(dataiter_list[n])
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])

            return images, labels

        
        w_global = model.get_weight()   

        rounds = 0
        
        not_participate_count_at_node = []
        participate_intervals_at_node = []

        for n in range(num_clients):
            not_participate_count_at_node.append(0)
            participate_intervals_at_node.append([])
        
        if algorithm == 'mifa' or algorithm == 'fedvarp':
            update_per_node = []
            for n in range(num_clients):
                update_per_node.append(torch.zeros(w_global.shape[0]).to('cpu'))
            update_per_node = torch.stack(update_per_node)
        if algorithm == 'fedvarp':
            update_all_avg = torch.zeros(w_global.shape[0]).to('cpu')

        while rounds < total_rounds:

            participation_prob_each_node = data_participation_each_node_per_round(p_each_client, rounds)
      
            worker_samplers = []
            for n in range(num_clients):
                worker_samplers.append(ClientSampler(participation_prob_each_node[n]))

            step_size_local_round = step_size_local/np.sqrt(rounds/10 + 1) 
            model.update_learning_rate(step_size_local_round)

            participation = np.array([False for i in range(num_clients)])
            accumulated = 0

            for n in range(num_clients):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participation[n] = True

                    model.assign_weight(w_global)
                    model.model.train()

                    for i in range(0, 10):
                        images, labels = sample_minibatch(n)

                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous()

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()                       
                        model.optimizer.step()


                    delta = model.get_weight()
                    delta -= w_global  

                    aggregation = None
                    if algorithm == 'fedknown':
                        delta /= worker_sampler.participation_prob
                        aggregation = 1/worker_sampler.participation_prob
                    elif algorithm == 'fedau':
                        if len(participate_intervals_at_node[n]) > 0:
                            aggregation = np.mean(participate_intervals_at_node[n])
                            delta *= aggregation
                    elif algorithm == 'fedvarp':
                        delta_new = copy.deepcopy(delta).to('cpu')
                        delta -= update_per_node[n].to(device)
                        update_per_node[n] = delta_new     
                    elif algorithm == 'mifa':
                        update_per_node[n] = delta.to('cpu')

                    participate_intervals_at_node[n].append(not_participate_count_at_node[n] + 1) 
                    not_participate_count_at_node[n] = 0

                else:
                    participation[n] = False
                    not_participate_count_at_node[n] += 1

                    if not_participate_count_at_node[n] >= 50:
                        participate_intervals_at_node[n].append(not_participate_count_at_node[n])
                        not_participate_count_at_node[n] = 0
                    delta = 0.0

                if algorithm != 'mifa':
                    if accumulated == 0:
                        w_accumulate = delta
                    else:
                        w_accumulate += delta

                    if algorithm != 'fedavg' and algorithm != 'fedvarp':
                        accumulated += 1
                    elif participation[n]:
                        accumulated += 1

            if algorithm != 'mifa':
                if accumulated > 0:
                    delta_a = torch.div(w_accumulate, torch.tensor(accumulated).to(device)).view(-1)
                    if algorithm == 'fedvarp':
                        delta_a += update_all_avg.to(device)
                        update_all_avg = torch.mean(update_per_node, 0)
                else:
                    delta_a = torch.zeros(w_global.shape[0]).to(device)
            else:
                delta_a = torch.mean(update_per_node, 0).to(device)

            w_global += torch.tensor(lr_global).to(device) * delta_a

            rounds += 1

            w_eval = w_global

            if rounds % eval_freq == 0:
                stat.collect_stat_eval(seed, rounds, model, data_train_loader, data_test_loader, w_eval)



        torch.cuda.empty_cache()

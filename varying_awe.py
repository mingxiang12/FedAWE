import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from config import *
from dataset.dataset import *
from statistic.collect_stat import stats_collector_bar as stats_collector
from util.util import data_participation_each_node, data_participation_each_node_per_round, DatasetSplit, ClientSampler, accumulation
import numpy as np
import random
from model.model import Model


def varying_dynamics_awe():
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
        dict_users, participation_prob_each_node_init = data_participation_each_node(data_train, num_clients)        

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
        w_local = torch.stack([w_global.detach().to('cpu') for i in range(num_clients)])

        rounds = 0
        not_participate_count_at_node = []

        for n in range(num_clients):
            not_participate_count_at_node.append(1)
   
        while rounds < total_rounds:

            participation_prob_each_node = data_participation_each_node_per_round(participation_prob_each_node_init, rounds)
      
            worker_samplers = []
            for n in range(num_clients):
                worker_samplers.append(ClientSampler(participation_prob_each_node[n]))

            step_size_local_round = step_size_local / np.sqrt(rounds/10+1) 
            model.update_learning_rate(step_size_local_round)

            participation = np.array([False for i in range(num_clients)])

            accumulated = 0.
            w_accumulate_local = 0.
            w_accumulate_delta = 0.

            for n in range(num_clients):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participation[n] = True
                
                    model.assign_weight(w_local[n].to('cuda'))
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
                
                    w_accumulate_local += w_local[n].to('cuda')
                    w_accumulate_delta += (model.get_weight() - w_local[n].to('cuda')) * not_participate_count_at_node[n]

                    accumulated += 1
                    not_participate_count_at_node[n] = 1

                
                else:
                    participation[n] = False
                    not_participate_count_at_node[n] += 1


            if accumulated > 0:
                w_global = accumulation(w_accumulate_local,w_accumulate_delta,accumulated,device,lr_global)
                accumulated = 0
            else:
                pass 

            for n in range(num_clients):
                if participation[n]:
                    w_local[n] =  w_global.detach().to('cpu') 

            rounds = rounds + 1

            if rounds % eval_freq == 0:
                stat.collect_stat_eval(seed, rounds, model, data_train_loader, data_test_loader, w_local, w_global)

        torch.cuda.empty_cache()

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from util.util import evaluate

class stats_collector:
    def __init__(self, prefix):
        os.makedirs('./csvs', exist_ok=True)
        self.file_name = './csvs/' + prefix + '.csv'

        with open(self.file_name, 'a') as f:
            f.write(
                'seed, rounds, train_loss, train_accuracy, test_accuracy\n')
            f.close()


    def collect_stat_eval(self, seed, rounds, model, train_data_loader, test_data_loader, w_global):
        w_eval = evaluate(w_global, device)
        
        train_loss, train_accuracy = model.accuracy(train_data_loader, w_eval, device, transform=transform_train_eval)
        _, test_accuracy = model.accuracy(test_data_loader, w_eval, device)

        print("seed", seed, "iteration", rounds,
              "train acc", train_accuracy, "test acc", test_accuracy)

        with open(self.file_name, 'a') as f:
            f.write(str(seed) + ',' + str(rounds) + ',' + str(train_loss) + ',' + str(train_accuracy) + ',' 
                    + str(test_accuracy) + '\n')
            f.close()


class stats_collector_bar:
    def __init__(self, prefix):
        os.makedirs('./csvs', exist_ok=True)
        self.file_name = './csvs/' + prefix + '.csv'


        with open(self.file_name, 'a') as f:
            f.write(
                'seed, rounds, train_loss, train_accuracy, test_accuracy \n')
            f.close()


    def collect_stat_eval(self, seed, rounds, model, train_data_loader, test_data_loader, w_local, w_global=None):
        w_eval = evaluate(w_global, device, w_local)
        
        train_loss, train_accuracy = model.accuracy(train_data_loader, w_eval, device, transform=transform_train_eval)
        _, test_accuracy = model.accuracy(test_data_loader, w_eval, device)
        
        print("seed", seed, "iteration", rounds,
              "train acc", train_accuracy, "test acc", test_accuracy)

        with open(self.file_name, 'a') as f:
            f.write(str(seed) + ',' + str(rounds) + ',' + str(train_loss) + ',' + str(train_accuracy) + ',' 
                    + str(test_accuracy) + '\n')
            f.close()
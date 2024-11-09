### Official Implementation of "Efficient Federated Learning against Heterogeneous and Non-stationary Client Unavailability"
---
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/08/Logo_for_Conference_on_Neural_Information_Processing_Systems.svg/1200px-Logo_for_Conference_on_Neural_Information_Processing_Systems.svg.png" width=200>

### Paper link:
Openreview: https://openreview.net/pdf?id=DLNOBJa7TM \
arXiv: https://arxiv.org/pdf/2409.17446

### Sources:
Our code is adapted from https://github.com/IBM/fedau, which is under MIT License.

### Supported algorithms: 
The supported algorithms are FedAvg over active clients (fedavg), FedAvg over all clients (fedall), FedAvg with known probabilities (fedknown), MIFA (mifa), FedAU (fedau), FedVARP (fedvarp) and FedAWE (fedawe).

### Supported datasets: 
The supported datasets are SVHN (svhn), CIFAR10 (cifar10) and CINIC10 (cinic10).

### Implementation examples:
python -m main --method fedavg --lr 0.05 --fluctuate 1 --dataset cifar10 --seeds 3,4,6 \
python -m main --method fedawe --lr 0.1 --fluctuate 1 --dataset cifar10 --seeds 3,4,6

### Unavailability dynamics:
The non-stationary dynamics are detailed as follows:

* Stationary: fluctuate 6;
* Non-stationary with sine:
    * fluctuate 1: $\gamma = 0.3$;
    * fluctuate 2: $\gamma = 0.2$;
    * fluctuate 3: $\gamma = 0.1$;
* Non-stationary with staircase: fluctuate 5;
* Non-stationary with interleaved sine: fluctuate 4.

### Visualizations:
In ``dynamic_visualization.ipynb'', we provide codes to reproduce the sampled dynamics trajectory.

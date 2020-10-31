### Adaptive Gradient Sparsification for Efficient Federated Learning: an Online Learning Approach

This repository includes source code for the paper P. Han, S. Wang, K. K. Leung, "Adaptive gradient sparsification for efficient federated learning: an online learning approach," IEEE International Conference on Distributed Computing Systems (ICDCS), Nov. 2020.

#### Getting Started

The code runs on Python 3 with Tensorflow version 1 (>=1.13). To install the dependencies, run
```
pip3 install -r requirements.txt
```

Then, download the datasets manually and put them into the `datasets` folder.
- For FEMNIST dataset，go to https://github.com/TalwalkarLab/leaf, clone the repository.

run

```cd ./data/femnist```

```./preprocess.sh -s niid --sf 0.05 -k 100 -t sample```

- For CIFAR-10 dataset, download the "CIFAR-10 binary version (suitable for C programs)" from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into `datasets/cifar-10-batches-bin`.

To test the code: 
- Set parameters in `config.py` 
- Run `simulation.py`

#### Code Structure

All configuration options are given in `config.py` which also explains the different setups that the code can run with.

The results are saved as CSV files in the `results` folder. 
The CSV files should be deleted before starting a new round of experiment.
Otherwise, the new results will be appended to the existing file.

Currently, the supported datasets are FEMNIST and CIFAR-10, and the supported model is CNN. The code can be extended to support other datasets and models too.  

#### Citation

When using this code for scientific publications, please kindly cite the above paper.

#### Third-Party Library

This code is derived from <https://github.com/IBM/adaptive-federated-learning>

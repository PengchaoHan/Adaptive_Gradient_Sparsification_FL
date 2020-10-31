This folder holds the datasets. The need to be downloaded separately. 

For FEMNIST dataset，go to https://github.com/TalwalkarLab/leaf, clone the repository.

run

```cd ./data/femnist```

```./preprocess.sh -s niid --sf 0.05 -k 100 -t sample```


For CIFAR-10 dataset, download the "CIFAR-10 binary version (suitable for C programs)" from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into the `cifar-10-batches-bin` folder.

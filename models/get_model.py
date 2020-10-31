import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_model(model_class_name):
    if model_class_name == 'ModelCNNEmnist':
        from models.cnn_emnist import ModelCNNEmnist
        return ModelCNNEmnist()
    elif model_class_name == 'ModelCNNCifar10':
        from models.cnn_cifar10 import ModelCNNCifar10
        return ModelCNNCifar10()
    else:
        raise Exception("Unknown model class name")

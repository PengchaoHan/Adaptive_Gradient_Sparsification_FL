from util.time_generation import TimeGeneration
import os


dataset_file_path = os.path.join(os.path.dirname(__file__), 'datasets')

results_file_path = os.path.join(os.path.dirname(__file__), 'results')

single_run_results_file_path = results_file_path + '/loss.csv'
single_run_results_file_path_k = results_file_path + '/weights.csv'

# datasets and models
dataset='EMNIST'
# dataset = 'CIFAR_10'
model_name = 'ModelCNNEmnist'
# model_name = 'ModelCNNCifar10'


n_nodes = 5  # number of clients
step_size = 0.01
batch_size = 32
total_data = 1000000

# Program stops if either max_time or max_iter is reached
max_time = 10000000000  # Total time budget in seconds
max_iter = 20000  # Maximum number of iterations to run

time_gen = TimeGeneration(1, 0.0, 1e-10, 0.0, 0.0, 0.0)
cost_communication_all_transmitted = 10  # time cost of full gradient communication

comp_method = 'FAB_top_K'  # Compression method: 'Always_send_all', 'FedAvg', 'U_top_K', 'FUB_top_K', 'PERIODIC_K', 'FAB_top_K'
k_init = 1000  # Upstream k parameter: None or a value, if None, use compression_ratio_init
compression_ratio_init = 0.5  # Initial compression ratio of 'U_top_K', 'FUB_top_K', 'PERIODIC_K', 'FAB_TOP_K'; k = dim_w * ratio
k_adaptive_method = 'NONE'  # Adaptive k method: 'NONE','Continuous_bandit', 'EXP3', 'OGD_SIGN', 'OGD_VALUE'


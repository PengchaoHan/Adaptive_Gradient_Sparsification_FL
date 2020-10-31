import numpy as np
import copy

from models.get_model import get_model
from util.collect_stat import CollectStatistics
from util.utils import get_indices_each_node_case
from util.sampling import MinibatchSampling
from data_reader.dataset import get_data
from control_algorithm.gtop_k import GTopK
from control_algorithm.periodic_k import PERIODIC_K
from control_algorithm.online_gradient_descent import ONLINE_GRADIENT_DESCENT
import random
from control_algorithm.mab_exp3 import EXP3
from control_algorithm.continuous_bandit import CONTINUOUS_BANDIT
import time

# Configurations are now in a separate config.py file
from config import *


"""
The above are configuration scripts, the "real" program starts here
"""
model = get_model(model_name)
if hasattr(model, 'create_graph'):
    model.create_graph(learning_rate=step_size)

sim = 0
MAX_CASE = 4
case = 1

if batch_size < total_data:   # Read all data once when using stochastic gradient descent
    train_image, train_label, test_image, test_label, train_label_orig = get_data(dataset, total_data,
                                                                                  dataset_file_path)

    indices_each_node_case = get_indices_each_node_case(n_nodes, MAX_CASE, train_label_orig)

stat = CollectStatistics(results_file_name=single_run_results_file_path)

data_size = np.zeros([MAX_CASE, n_nodes])  # Data size for different cases and different n
for case in range(MAX_CASE):
    for n in range(n_nodes):
        data_size[case][n] = len(indices_each_node_case[case][n])

unique_labels = np.unique(train_label,axis=0).tolist()

dim_w = model.get_weight_dimension(train_image, train_label)
w_global_init = model.get_init_weight(dim_w, rand_seed=sim)
w_global = copy.deepcopy(w_global_init)
random.seed(sim)

w_global_prev = copy.deepcopy(w_global)

total_time = 0      # Actual total time, where use_fixed_averaging_slots has no effect
sampler_list = []
train_indices_list = []
w_list = []

# -------------------------------------- initialize each client
for n in range(0, n_nodes):
    indices_this_node = indices_each_node_case[case][n]

    if batch_size >= total_data:
        sampler = None
        train_indices = indices_this_node
    else:
        if batch_size > len(indices_this_node):
            sampler = MinibatchSampling(indices_this_node, len(indices_this_node), sim)  # For nodes whose sample length are less than batch size
        else:
            sampler = MinibatchSampling(indices_this_node, batch_size, sim)
        train_indices = None  # To be defined later

    sampler_list.append(sampler)
    train_indices_list.append(train_indices)
    w_list.append(copy.deepcopy(w_global_init))

grad_array = np.zeros([n_nodes, dim_w])

# -------------------------------------- initialize compression method
print("compression method:", comp_method)
print("adaptive method:", k_adaptive_method)

tau_setup = 1
if comp_method == 'Always_send_all':
    k = dim_w
elif comp_method == 'FedAvg':
    tau_setup = int(np.ceil(dim_w / k_init / 2))
    print('FedAvg:', tau_setup)
elif comp_method == 'U_top_K' or comp_method == 'FAB_top_K':
    g_top_k = GTopK(dim_w)
    if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
        loss = np.zeros([n_nodes])
        loss_prev = np.zeros([n_nodes])
        loss_aux = np.zeros([n_nodes])
        if k_init is None:
            k = int(np.floor(compression_ratio_init * dim_w))  # k_up
        else:
            k = k_init
        k_prev = k
        k_down = k
        k_aux = int(np.ceil(k * 0.9))
        k_aux_down = k_aux
        ogd = ONLINE_GRADIENT_DESCENT(int(np.round(dim_w * 0.002)), dim_w)  # Min. cannot be 0, use 0.2% of all weights as min.
        w_global_aux = copy.deepcopy(w_global)
    elif k_adaptive_method == 'EXP3':
        loss = np.zeros([n_nodes])
        loss_prev = np.zeros([n_nodes])
        k_list = [i+10 for i in range(dim_w-10)]
        T = 5780  # 1000
        ep3 = EXP3(dim_w, T, k_list)
        k = ep3.pick_choice()  # Initial value
        prob = copy.deepcopy(ep3.prob)
        k_down = k
    elif k_adaptive_method == 'Continuous_bandit':
        loss = np.zeros([n_nodes])
        loss_prev = np.zeros([n_nodes])
        action_dimension = 1
        max_k = np.ones(action_dimension)*dim_w
        min_k = np.zeros(action_dimension)+10
        T = 5780  # 1000
        cb = CONTINUOUS_BANDIT(action_dimension, max_k, min_k, T)
        k = cb.get_initial_action()  # Get initial compression ratio
        k_down = k
    elif k_adaptive_method == 'NONE':
        if k_init is None:
            k = int(np.floor(compression_ratio_init * dim_w))  # k_up
        else:
            k = k_init
        k_prev = k
        k_down = k
    else:
        raise Exception("Unknown adaptive compression name")
elif comp_method == 'FUB_top_K':
    g_top_k = GTopK(dim_w)
    if k_init is None:
        k = int(np.floor(compression_ratio_init * dim_w))  # k_up
    else:
        k = k_init
    k_prev = k
elif comp_method == 'PERIODIC_K':
    if k_init is None:
        k = int(np.floor(compression_ratio_init * dim_w))  # k_up
    else:
        k = k_init
    k_prev = k
    pk = PERIODIC_K(k, dim_w)
else:
    raise Exception("Unknown compression name")

num_iter = 0

expr_start_time = time.time()

# Loop for multiple rounds of local iterations + global aggregation
while True:

    num_iter = num_iter + tau_setup
    if (comp_method == 'U_top_K' or comp_method == 'FAB_top_K') and (k_adaptive_method == 'OGD_SIGN'or k_adaptive_method == 'OGD_VALUE'):
        w_global_prev_2 = copy.deepcopy(w_global_prev)
    w_global_prev = copy.deepcopy(w_global)

    train_indices_current_list = []
    for n in range(0, n_nodes):
        train_indices = train_indices_list[n]
        use_consecutive_training = False
        if tau_setup > 1:
                use_consecutive_training = True

        for i in range(0, tau_setup):

            if batch_size < total_data:
                sample_indices = sampler_list[n].get_next_batch()
                train_indices = sample_indices

            train_indices_current_list.append(train_indices)

            if use_consecutive_training:
                model.run_one_step_consecutive_training(train_image, train_label, train_indices)
            else:
                grad = model.gradient(train_image, train_label, w_list[n], train_indices)

                grad_array[n] += grad
                if tau_setup > 1:
                    w_list[n] = w_list[n] - step_size * grad  # For multiple tau

        if use_consecutive_training:
            w_list[n] = model.end_consecutive_training_and_get_weights()

    if comp_method == 'U_top_K':
        g_global, mask, local_mask = g_top_k.general_top_k(k, k_down, grad_array, n_nodes, data_size[case])
        w_global = w_global_prev - step_size * g_global
        if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
            g_global_aux, _, _ = g_top_k.general_top_k(k_aux, k_aux_down, grad_array, n_nodes, data_size[case])
            w_global_aux = w_global_prev - step_size * g_global_aux
    elif comp_method == 'FAB_top_K':
        import time
        g_global, mask, local_mask = g_top_k.fairness_aware_top_k(k, k_down, grad_array, n_nodes, data_size[case])
        w_global = w_global_prev - step_size * g_global
        if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
            g_global_aux, _, _ = g_top_k.fairness_aware_top_k(k_aux, k_aux_down, grad_array, n_nodes,
                                                       data_size[case])
            w_global_aux = w_global_prev - step_size * g_global_aux
    elif comp_method == "FUB_top_K":
        g_global, mask, local_mask = g_top_k.global_top_k(k, grad_array, n_nodes, data_size[case])
        w_global = w_global_prev - step_size * g_global
    elif comp_method == 'PERIODIC_K':
        mask = pk.generate_mask()
        grad_array_ = np.multiply(grad_array, mask)
        g_global = np.dot(data_size[case], grad_array_)/sum(data_size[case])
        w_global = w_global_prev - step_size * g_global
        mask_r = np.zeros(dim_w)
        mask_r[mask == 0] = 1
        grad_array = np.multiply(grad_array, mask_r)  # Update gradient residual
    else:
        if use_consecutive_training:
            w_global = np.dot(data_size[case], w_list) / sum(data_size[case])
        else:
            g_global = np.dot(data_size[case], grad_array) / sum(data_size[case])
            w_global = w_global_prev - step_size * g_global
            grad_array = np.zeros([n_nodes, dim_w])

    # ----------------------- synchronize weights among all clients
    if True in np.isnan(w_global):
        print('*** w_global is NaN, using previous value')
        w_global = copy.deepcopy(w_global_prev)   # If current w_global contains NaN value, use previous w_global

    for n in range(0, n_nodes):
        w_list[n] = copy.deepcopy(w_global)

    # ------------------------ find the next k
    if comp_method == 'U_top_K' or comp_method == 'FAB_top_K':
        if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
            for n in range(0, n_nodes):
                train_indices = train_indices_current_list[n]
                tmp_indices = [random.choice(train_indices)]
                loss[n] = model.loss(train_image, train_label, w_list[n], tmp_indices)
                loss_prev[n] = model.loss(train_image, train_label, w_global_prev, tmp_indices)  # Get the loss base on new mini-batch
                loss_aux[n] = model.loss(train_image, train_label, w_global_aux, tmp_indices)

            global_loss = np.sum(np.multiply(loss, data_size[case])) / sum(data_size[case])  # Get the global loss in aggregator by collecting local losses
            global_loss_prev = np.sum(np.multiply(loss_prev, data_size[case])) / sum(data_size[case])  # Get the global loss base on new mini-batch
            global_loss_aux = np.sum(np.multiply(loss_aux, data_size[case])) / sum(data_size[case])

            k_prev = k

            cost = 0
            if num_iter > 2:
                cost = comp_time + comm_time
                if np.isnan(global_loss) or np.isnan(global_loss_aux):
                    print('*** loss is NaN!')
                    w_global = copy.deepcopy(w_global_prev)
                    w_global_prev = copy.deepcopy(w_global_prev_2)
                    cost_aux = None

                elif global_loss_prev - global_loss > 0 and global_loss_prev - global_loss_aux > 0:
                    cost_aux = (comp_time + comm_time_aux) * (global_loss_prev - global_loss) / (global_loss_prev - global_loss_aux)
                    if np.isnan(cost_aux):
                        cost_aux = None
                else:
                    cost_aux = None
                print('global_loss_prev:', global_loss_prev)
                print('global_loss:', global_loss)
                if k_adaptive_method == 'OGD_SIGN':
                    k, k_aux = ogd.tuning_k_grad_sign(k, k_aux, cost, cost_aux, num_iter)
                else:  # 'VALUE'
                    k, k_aux = ogd.tuning_k_grad_value(k, k_aux, cost, cost_aux, num_iter)
                k_down = k
                k_aux_down = k_aux

        elif k_adaptive_method == 'EXP3':
            for n in range(0, n_nodes):
                train_indices = train_indices_current_list[n]
                tmp_indices = [random.choice(train_indices)]
                loss[n] = model.loss(train_image, train_label, w_list[n], tmp_indices)
                loss_prev[n] = model.loss(train_image, train_label, w_global_prev, tmp_indices)  # Get the loss base on new mini-batch

            global_loss = np.sum(np.multiply(loss, data_size[case])) / sum(data_size[case])  # Get the global loss in aggregator by collecting local losses
            global_loss_prev = np.sum(np.multiply(loss_prev, data_size[case])) / sum(data_size[case])  # Get the global loss base on new mini-batch

            reward = 0
            k_prev = k
            if num_iter > 2:
                reward = (comp_time + comm_time) / (global_loss - global_loss_prev)  # To be maximized
                k, prob = ep3.step(k_prev, reward)  # Get next k
                print('EXP3 rewards:' + str(reward) + ', next k:' + str(k))
                k_down = k

        elif k_adaptive_method == 'Continuous_bandit':
            for n in range(0, n_nodes):
                train_indices = train_indices_current_list[n]
                tmp_indices = [random.choice(train_indices)]
                loss[n] = model.loss(train_image, train_label, w_list[n], tmp_indices)
                loss_prev[n] = model.loss(train_image, train_label, w_global_prev, tmp_indices)  # Get the loss base on new mini-batch

            global_loss = np.sum(np.multiply(loss, data_size[case])) / sum(data_size[case])  # Get the global loss in aggregator by collecting local losses
            global_loss_prev = np.sum(np.multiply(loss_prev, data_size[case])) / sum(data_size[case])  # Get the global loss base on new mini-batch

            cost = 0
            k_prev = k
            if num_iter > 2:
                if global_loss_prev - global_loss == 0:
                    cost = None
                else:
                    cost = (comp_time + comm_time) / (global_loss_prev - global_loss)  # To be minimized
                k = cb.get_next_action(cost)
                print('Continuous_bandit cost:' + str(cost) + ', next k:' + str(k))
                k_down = k

    # ------------------------ accumulate residual gradients and collect information
    if comp_method == "U_top_K" or comp_method == 'FAB_top_K' or comp_method == "FUB_top_K":  # Update residuals
        for i in range(n_nodes):
            local_mask[i][mask == 0] = 0
        local_mask_r = np.zeros([n_nodes, dim_w])
        local_mask_r[local_mask == 0] = 1
        grad_array = np.multiply(grad_array, local_mask_r)  # Update gradient residual

        k_down_actual = np.sum(g_global != 0)  # Number of gradient for download transmission
        if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
            k_aux_down_actual = np.sum(g_global_aux != 0)
        weights = [np.sum(local_mask[i]) for i in range(n_nodes)]  # Number of gradients of nodes that used in the global aggregating
        accu_local_mask = np.zeros(dim_w)
        for i in range(n_nodes):
            accu_local_mask += local_mask[i]
        accu_local_mask[accu_local_mask == 1] = 0
        accu_local_mask[accu_local_mask > 1] = 1
        num_aggregated = np.sum(accu_local_mask)  # Number of weights that aggregated multiple nodes
        stat.collect_stat_end_global_round_weights(case, num_iter, k_prev, weights, k_down_actual, num_aggregated)
    elif comp_method == "PERIODIC_K":
        k_down_actual = np.sum(g_global != 0)  # Number of gradient for download transmission
        weights = np.ones(n_nodes) * np.sum(mask)  # Number of gradients of nodes that used in the global aggregating
        num_aggregated = np.sum(mask)  # Number of weights that aggregated multiple nodes
        stat.collect_stat_end_global_round_weights(case, num_iter, k, weights, k_down_actual, num_aggregated)

    # ------------------------ obtain time cost
    if isinstance(time_gen, (list,)):
        t_g = time_gen[case]
    else:
        t_g = time_gen
    it_each_local = max(0.00000001, np.sum(t_g.get_local(tau_setup)) / tau_setup)
    it_each_global = t_g.get_global(1)[0]
    #Compute number of iterations is current slot
    comp_time = it_each_local * tau_setup + it_each_global

    comm_time = 0
    if comp_method == 'U_top_K' or comp_method == 'FAB_top_K':
        comm_time = cost_communication_all_transmitted / dim_w / 2 * (k * 2 + k_down_actual * 2)  # *2 for both values and indices, Compress for both upstream and downstream
        if k_adaptive_method == 'OGD_SIGN' or k_adaptive_method == 'OGD_VALUE':
            comm_time_aux = cost_communication_all_transmitted / dim_w / 2 * (k_aux * 2 + k_aux_down_actual * 2)
            comp_time *= (1 + 1.0/batch_size)  # To take into account for additional loss computation and top k
        elif k_adaptive_method == 'EP3' or k_adaptive_method == 'Continuous_bandit':
            comp_time *= (1 + 1.0 / batch_size)
    elif comp_method == 'PERIODIC_K' or comp_method == "FUB_top_K":
        comm_time = cost_communication_all_transmitted / dim_w * k * 2  # The cost of random seed is very small
    else:  # 'FedAvg' or 'Always_send_all'
        comm_time = cost_communication_all_transmitted * tau_setup  # General case

    total_time += comp_time + comm_time
    expr_cur_time = time.time() - expr_start_time # experimental time overhead

    if (num_iter-1) % 20 == 0:
        stat.collect_stat_end_local_round(case, num_iter, model,
                                       train_image, train_label, test_image, test_label, w_global,
                                       total_time, expr_cur_time, k)

    if total_time >= max_time:
        break
    if num_iter >= max_iter:
        break


print("number of iterations:", num_iter)

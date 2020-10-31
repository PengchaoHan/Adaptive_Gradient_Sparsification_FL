import numpy as np
import copy
import tensorflow as tf


class GTopK:  # Use tensorflow
    def __init__(self, dim):
        self.dim = dim
        self.value = tf.placeholder(tf.float32,shape = [None,self.dim])
        self.k = tf.placeholder(tf.int32)
        self.tk_op = tf.nn.top_k(self.value, self.k)
        self.sess = tf.Session()

    def top_k(self, k, values, batch_size): # Extract top k of absolute values and corresponding mask, batch_size is the number of 1-D array to compute topk
        if batch_size == 1:
            mask = np.zeros([1, self.dim])
            v_ = np.array([values]).reshape(-1)
            _, i = self.sess.run(self.tk_op, feed_dict={self.value: np.abs(values).reshape(1, -1), self.k: k})
            mask[0][i] = 1
            v_[mask[0] == 0] = 0
        else:
            mask = np.zeros([batch_size, self.dim])
            v_ = copy.deepcopy(values)
            _, i = self.sess.run(self.tk_op, feed_dict={self.value: np.abs(values), self.k: k})
            for n in range(batch_size):
                mask[n][i[n]] = 1
                v_[n][mask[n] == 0] = 0
        return v_, mask

    def general_top_k(self, k_up, k_down, grad_array, n_nodes, data_size):
        g_global, local_mask = self.top_k(k_up,grad_array,n_nodes)  # Upstream topk
        if n_nodes>1:
            g_global = np.dot(data_size, g_global) / sum(data_size)
        g_global, mask = self.top_k(k_down,g_global,1)  # Downstream topk
        mask = mask.reshape(-1)
        return g_global, mask, local_mask

    def global_top_k(self, k, grad_array, n_nodes, data_size):  # Extract global top k values and corresponding mask
        g_array_, local_mask = self.top_k(k,grad_array,n_nodes)  # Upstream topk
        nRound = int(np.ceil(np.log2(n_nodes)))
        num_g_this_round = n_nodes
        g_this_round = copy.deepcopy(g_array_)
        data_size_this_round = copy.deepcopy(data_size)
        mask = copy.deepcopy(local_mask[0])
        for i in range(nRound):
            num_g_next_round = int(np.ceil(num_g_this_round / 2))
            g_ = np.zeros([num_g_next_round, self.dim])
            data_size_this_round_ = []
            for n in range(num_g_this_round):
                if n%2 == 1:  # 1,3,5,7
                    gs = (g_this_round[n] * data_size_this_round[n] + g_this_round[n-1] * data_size_this_round[n-1])/(data_size_this_round[n]+data_size_this_round[n-1])
                    g_[int(np.floor(n / 2))], mask = self.top_k(k, gs, 1)
                    mask = mask.reshape(-1)
                    for j in range(min(np.power(2,i+1),n_nodes)):
                        n_idx = int((n-1)/2*np.power(2,i+1)+j)
                        if(n_idx < n_nodes):
                            local_mask[n_idx][mask == 0] = 0
                    data_size_this_round_.append(data_size_this_round[n]+data_size_this_round[n-1])
                if n == num_g_this_round-1 and n%2 == 0:  # For last g that cannot be merged
                    g_[int(np.floor((n + 1) / 2))] = g_this_round[n]  # g_[int(np.floor((n+1)/2))], mask_ = self.top_k(k, g_this_round[n])
                    data_size_this_round_.append(data_size_this_round[n])
            g_this_round = copy.deepcopy(g_)
            num_g_this_round = num_g_next_round
            data_size_this_round = copy.deepcopy(data_size_this_round_)
        g_global = copy.deepcopy(g_this_round[0])
        if n_nodes == 1:
            g_global = copy.deepcopy(g_this_round)
        mask = mask.reshape(-1)
        return g_global, mask, local_mask

    def fairness_aware_top_k(self, k_up, k_down, grad_array, n_nodes, data_size):
        # Local (Upstream) TopK
        grad_, local_mask = self.top_k(k_up, grad_array, n_nodes)
        # Find mask
        if n_nodes == 1:
            _, sorted_indices = self.sess.run(self.tk_op, feed_dict={self.value: np.abs(grad_).reshape(1, -1), self.k: k_down})  # Sort absolute grad_
        else:
            _, sorted_indices = self.sess.run(self.tk_op, feed_dict={self.value: np.abs(grad_), self.k: k_down})  # Sort absolute grad_
        num_k = 0
        mask=np.zeros(self.dim)
        for i in range(0,k_down):
            if num_k == k_down:
                break
            if n_nodes == 1:
                grad_value_k = [grad_[int(sorted_indices[n][i])] for n in range(n_nodes)]  # Values of jth largest grad_ of n nodes
            else:
                grad_value_k = [grad_[n][int(sorted_indices[n][i])] for n in range(n_nodes)]  # Values of jth largest grad_ of n nodes
            sorted_indices_node = np.argsort(np.abs(grad_value_k))  # Sort the jth largest values

            for n in range(n_nodes):
                if mask[sorted_indices[sorted_indices_node[n_nodes-1-n]][i]] == 0:
                    mask[sorted_indices[sorted_indices_node[n_nodes-1-n]][i]] = 1
                    num_k += 1
                    if num_k == k_down:
                        break

        # Aggregating
        g_global = np.multiply(grad_, mask)
        if n_nodes > 1:
            g_global = np.dot(data_size, g_global) / sum(data_size)
        local_mask = np.multiply(local_mask, mask)
        return g_global, mask, local_mask

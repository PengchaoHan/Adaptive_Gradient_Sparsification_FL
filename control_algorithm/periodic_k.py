import numpy as np


class PERIODIC_K():

    def __init__(self, k, dim):  # p = ceil(dim/k) *2-1
        self.k = k
        self.dim = dim
        self.time_flag = np.zeros(self.dim)

    def generate_mask(self):  # Generate one mask at a time
        mask = np.zeros(self.dim)
        res_len = int(self.dim - sum(self.time_flag))
        if res_len == self.k:
            mask[self.time_flag == 0] = 1
            residual_indices1 = np.argwhere(self.time_flag == 0)
            self.time_flag = np.zeros(self.dim)
            self.time_flag[residual_indices1] = 1
        elif res_len < self.k:
            randv = np.random.rand(int(sum(self.time_flag)))
            sorted_indices = np.argsort(randv)
            residual_indices = np.argwhere(self.time_flag == 1)
            mask[residual_indices[sorted_indices[-(self.k-res_len):]]] = 1

            mask[self.time_flag == 0] = 1
            residual_indices1 = np.argwhere(self.time_flag == 0)

            self.time_flag = np.zeros(self.dim)
            self.time_flag[residual_indices1] = 1
            self.time_flag[residual_indices[sorted_indices[-(self.k-res_len):]]] = 1
        else:
            randv = np.random.rand(res_len)
            sorted_indices = np.argsort(randv)
            residual_indices = np.argwhere(self.time_flag == 0)
            mask[residual_indices[sorted_indices[-self.k:]]] = 1
            self.time_flag[residual_indices[sorted_indices[-self.k:]]] = 1

        return mask


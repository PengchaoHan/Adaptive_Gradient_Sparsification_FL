import numpy as np
import random

class ONLINE_GRADIENT_DESCENT:
    def __init__(self, k_min,k_max):
        self.k_min_orig = k_min
        self.k_max_orig = k_max
        self.k_min = k_min
        self.k_max = k_max
        self.d = k_max - k_min
        self.delta = 0.1   #NEW: min. value for k_aux_next is self.k_min * self.delta   # Tunable parameter, TODO: move value assignment to outside of class

        self.min_max_update_window = 20   # Tunable parameter, TODO: move value assignment to outside of class
        self.alpha = 1.5   # Tunable parameter, TODO: move value assignment to outside of class
        self.min_max_update_count = 0
        self.k_min_window = self.k_max_orig   #Note inverse min/max assignment
        self.k_max_window = self.k_min_orig   #Note inverse min/max assignment
        self.reference_iter = 0
        self.m_prev = 0

        self.timer = 0
        self.grad_value_prev = 0

    def tuning_k_grad_sign(self,k,k_aux,cost,cost_aux,time):  # k>k_aux
        eta = self.d / np.sqrt(2*(time - self.reference_iter))
        k_unchanged_due_to_cost_none = False
        if cost_aux is None:
            # if k > eta and self.timer <= 5:
            if self.timer <= 10:
                k_next = k
                self.timer += 1
                k_unchanged_due_to_cost_none = True
            else:
                self.timer = 0
                k_next = self.stochasitic_rouding(k + eta)  # When any loss of k and k_aux increases
            print('cost_aux is None')
        else:
            self.timer = 0
            k_next = self.stochasitic_rouding(k - eta * np.sign((cost - cost_aux)/(k-k_aux)))  # To minimize cost
        print('k_next:', k_next)

        if k_next < self.k_min:
            k_next = self.k_min
        elif k_next > self.k_max:
            k_next = self.k_max

        k_aux_next = self.stochasitic_rouding(k_next - eta / 2.0)
        if k_aux_next < self.k_min * self.delta:
            k_aux_next = int(np.ceil(self.k_min * self.delta))

        if k_aux_next >= k_next:
            k_aux_next = k_next - 1

        if not k_unchanged_due_to_cost_none:
            self.k_min_window = min(self.k_min_window, k_next)
            self.k_max_window = max(self.k_max_window, k_next)
            self.min_max_update_count += 1

        if self.min_max_update_count >= self.min_max_update_window:
            self.min_max_update_count = 0
            k_min_window_change = self.k_min_window / self.alpha
            k_max_window_change = self.k_max_window * self.alpha
            k_min_window_change = int(np.round(max(k_min_window_change, self.k_min_orig)))
            k_max_window_change = int(np.round(min(k_max_window_change, self.k_max_orig)))
            b_new = k_max_window_change - k_min_window_change
            b_orig = self.k_max - self.k_min
            m_current = time - self.reference_iter

            if b_new > 0 and m_current >= self.m_prev and b_orig + b_new <= b_orig * np.sqrt(2):
            # if b_new > 0:
                self.k_min = k_min_window_change
                self.k_max = k_max_window_change
                self.d = self.k_max - self.k_min
                self.reference_iter = time
                self.m_prev = m_current
                print('******** New k_min:', self.k_min, 'new k_max:', self.k_max)
            else:
                # # print('float(b_orig) * float(b_orig) + 4.0 * float(self.k_min_window) * float(self.k_max_window):',
                # #       float(b_orig) * float(b_orig) + 4.0 * float(self.k_min_window) * float(self.k_max_window))
                # alpha_new = (float(b_orig) + np.sqrt(float(b_orig) * float(b_orig) + 4.0 * float(self.k_min_window) * float(self.k_max_window))) / (2.0 * float(self.k_max_window))
                # k_min_window_remain = self.k_min_window / alpha_new
                # k_max_window_remain = self.k_max_window * alpha_new
                # if k_min_window_remain < self.k_min_orig:
                #     k_max_window_remain += self.k_min_orig - k_min_window_remain
                #     k_min_window_remain = self.k_min_orig
                # elif k_max_window_remain > self.k_max_orig:
                #     k_min_window_remain -= k_max_window_remain - self.k_max_orig
                #     k_max_window_remain = self.k_max_orig
                # k_min_window_remain = int(np.round(max(k_min_window_remain, self.k_min_orig)))
                # k_max_window_remain = int(np.round(min(k_max_window_remain, self.k_max_orig)))
                #
                # self.k_min = k_min_window_remain
                # self.k_max = k_max_window_remain

                print('******** Same range - New k_min_window:', self.k_min, 'new k_max_window:', self.k_max, 'b_orig:', b_orig, 'b_new:', self.k_max - self.k_min)
                print('m_current:', m_current, 'self.m_prev:', self.m_prev)
                print('b_orig * np.sqrt(m_current) + b_new * np.sqrt(self.min_max_update_window) =', b_orig * np.sqrt(m_current) + b_new * np.sqrt(self.min_max_update_window))
                print('b_orig * np.sqrt(m_current + self.min_max_update_window) =', b_orig * np.sqrt(m_current + self.min_max_update_window))

            self.k_min_window = self.k_max_orig  # Note inverse min/max assignment
            self.k_max_window = self.k_min_orig  # Note inverse min/max assignment

        return k_next, k_aux_next

    def tuning_k_grad_value(self,k,k_aux,cost,cost_aux,time):  # k>k_aux
        eta = self.d / np.sqrt(2*time)
        grad_value = None
        if cost_aux is None:
            # if k > eta and self.timer <= 5:
            if self.timer <= 10:
                k_next = k
                self.timer += 1
            else:
                self.timer = 0
                k_next = self.stochasitic_rouding(k + eta * self.grad_value_prev)  # When any loss of k and k_aux increases
            print('cost_aux is None')
        else:
            self.timer = 0
            grad_value = (cost - cost_aux)/(k-k_aux)
            k_next = self.stochasitic_rouding(k - eta * grad_value)  # To minimize cost
            self.grad_value_prev = grad_value
        print('k_next:', k_next)

        if k_next < self.k_min:
            k_next = self.k_min
        elif k_next > self.k_max:
            k_next = self.k_max

        if grad_value is None:
            grad_value = self.grad_value_prev
        k_aux_next = self.stochasitic_rouding(k_next - eta * np.abs(grad_value) / 2.0)
        if k_aux_next < self.k_min * self.delta:
            k_aux_next = int(np.ceil(self.k_min * self.delta))

        if k_aux_next >= k_next:
            k_aux_next = k_next - 1

        return k_next, k_aux_next

    def stochasitic_rouding(self, x):
        floor_x = int(np.floor(x))
        prob = random.random()
        if prob < x - floor_x:
            x = floor_x + 1
        else:
            x = floor_x
        return x


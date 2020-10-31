import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *


class CollectStatistics:
    def __init__(self, results_file_name=os.path.dirname(__file__)+'/results.csv'):
        self.results_file_name = results_file_name

        with open(results_file_name, 'a') as f:
            f.write(
                'case,num_iter,tValue,lossValue,predictionAccuracy,exprTime, k\n')
            f.close()

        with open(single_run_results_file_path_k, 'a') as f:
            f.write(
                'case,num_iter,k_up,weights,k_down,num_aggregated\n')
            f.close()

    def collect_stat_end_local_round(self, case, num_iter, model, train_image,
                                     train_label, test_image, test_label, w_global, total_time_recomputed, experimental_time=None, k=None,):
            loss_value = model.loss(train_image, train_label, w_global)
            prediction_accuracy = model.accuracy(test_image, test_label, w_global)

            print("***** lossValue: " + str(loss_value))

            with open(self.results_file_name, 'a') as f:
                f.write(str(case) + ',' + str(num_iter) + ',' + str(total_time_recomputed) + ',' + str(loss_value) + ','
                        + str(prediction_accuracy) + ',' + str(experimental_time) + ',' + str(k) + '\n')
                f.close()

    def collect_stat_end_global_round_weights(self,case,num_iter,k_up, weights,k_down, num_aggregated):
        with open(single_run_results_file_path_k, 'a') as f:
            f.write(str(case) + ',' + str(num_iter) + ',' + str(k_up) + ',' + str(weights) + ',' + str(k_down) + ','+ str(num_aggregated) +'\n')
            f.close()

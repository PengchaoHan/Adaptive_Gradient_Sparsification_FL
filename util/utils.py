import numpy as np
import math

def get_index_from_one_hot_label(label):
    for i in range(0, len(label)):
        if (label[i] == 1):
            return [i]

def get_one_hot_from_label_index(label, number_of_labels=10):
    one_hot = np.zeros(number_of_labels)
    one_hot[label] = 1
    return one_hot

def get_indices_each_node_case(n_nodes, maxCase, label_list):
    indexesEachNodeCase = []


    for i in range(0, maxCase):
        indexesEachNodeCase.append([])

    for i in range(0, n_nodes):
        for j in range(0, maxCase):
            indexesEachNodeCase[j].append([])

    # indexesEachNode is a big list that contains N-number of sublists. Sublist n contains the indexes that should be assigned to node n

    minLabel = min(label_list)
    maxLabel = max(label_list)
    numLabels = maxLabel - minLabel + 1

    for i in range(0, len(label_list)):

        # case 1

        #randomperm = np.random.permutation(len(train))
        #indexesEachNodeCase[0][(i % n_nodes)].append(randomperm[i])
        indexesEachNodeCase[0][(i % n_nodes)].append(i)

        # case 2

        #indexesEachNodeCase[1][(train[i][2][0] % n_nodes)].append(i)
        tmp_target_node=int((label_list[i]-minLabel) %n_nodes)


        #indexesEachNodeCase[1][(label_list[i] % n_nodes)].append(i)

        if n_nodes>numLabels:
            tmpMinIndex=0
            tmpMinVal=math.inf
            for n in range(0,n_nodes):
                if (n)%numLabels==tmp_target_node and len(indexesEachNodeCase[1][n])<tmpMinVal:
                    tmpMinVal=len(indexesEachNodeCase[1][n])
                    tmpMinIndex=n
            tmp_target_node=tmpMinIndex

        indexesEachNodeCase[1][tmp_target_node].append(i)



        # case 3
        for n in range(0, n_nodes):
            indexesEachNodeCase[2][n].append(i)

        # case 4

        tmp = int(np.ceil(min(n_nodes,numLabels)/2))

        #if train[i][2][0] < 5:   # NOTE: The number 5 only works for MNIST data
        if label_list[i] < (minLabel+maxLabel)/2:
            #tmp_target_node = train[i][2][0] % tmp
            tmp_target_node=i %tmp

        elif n_nodes > 1:
            tmp_target_node = int(((label_list[i] -minLabel) % (min(n_nodes,numLabels)-tmp))+tmp)


        if n_nodes>numLabels:
            tmpMinIndex=0
            tmpMinVal=math.inf
            for n in range(0,n_nodes):
                if (n)%numLabels==tmp_target_node and len(indexesEachNodeCase[3][n])<tmpMinVal:
                    tmpMinVal=len(indexesEachNodeCase[3][n])
                    tmpMinIndex=n
            tmp_target_node=tmpMinIndex

        indexesEachNodeCase[3][tmp_target_node].append(i)


    return indexesEachNodeCase

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import get_index_from_one_hot_label, get_one_hot_from_label_index

def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None, use_test_client_to_data_dict=False):
    test_client_to_data_dict = None

    if dataset=='EMNIST':

        from data_reader.emnist_extractor import read_data
        clients, groups, train_data, test_client_to_data_dict = read_data(dataset_file_path +'/femnist/train',dataset_file_path +'/femnist/test' )
        #print('clients',clients)
        #print('groups',groups)

        list_train_keys=list(train_data.keys())
        #print('list train keys',len(list_train_keys))
        train_image=[]
        train_label=[]
        train_label_orig=[]



        for i in range(0,len(list_train_keys)):
            # note: each time we append a list
            train_image+= train_data[list_train_keys[i]]["x"]
            train_label+= train_data[list_train_keys[i]]["y"]
            #print('client',i)
            #print(len(train_data[list_train_keys[i]]["y"]))
            for j in range(0, len(train_data[list_train_keys[i]]["x"])):
                train_label_orig.append(i)

        ##print('train orig',train_label_orig)
        #print('training orgin', len(train_label_orig))
        #print('train image shape')
        #print(np.array(train_image).shape)
        for i in range(0,len(train_label)):
            train_label[i]=get_one_hot_from_label_index(train_label[i],62)

        test_image = []
        test_label = []
        test_label_orig=[]

        list_test_keys=list(test_client_to_data_dict.keys())
        #print('test')
        #print(list(test_data.keys()))
        #print('len test',len(list(test_data.keys())))
        for i in range(0,len(list_test_keys)):
            test_image+= test_client_to_data_dict[list_test_keys[i]]["x"]
            test_label+= test_client_to_data_dict[list_test_keys[i]]["y"]
            for j in range(0, len(test_client_to_data_dict[list_test_keys[i]]["x"])):
                test_label_orig.append(i)


        for i in range(0,len(test_label)):
            test_label[i]=get_one_hot_from_label_index(test_label[i],62)

        #print('test shape')
        #print(np.array(test_image).shape)
        #print(np.array(test_label).shape)

        #print('test label or', len(test_label))
        #print('test label orgine',test_label_orig

    elif dataset=='CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

    else:
        raise Exception('Unknown dataset name.')

    if use_test_client_to_data_dict:
        return train_image, train_label, test_image, test_label, train_label_orig, test_client_to_data_dict
    else:
        return train_image, train_label, test_image, test_label, train_label_orig


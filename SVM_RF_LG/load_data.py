import pandas as pd
import numpy as np
import os

# SVM/RF for stock trend prediction
# Input：T*F  Output： T*1

# ACL18 data
# data_path = '../data/ACL18/ourpped'
# KDD17 data
# data_path = '../data/KDD17/ourpped'

def files_name(path):
    filesname_list = []
    for i in range(len(path)):
        (filepath, tempfilename) = os.path.split(path[i])
        (filesname, extension) = os.path.splitext(tempfilename)
        filesname_list.append(filesname)
    # print("stock_id list:",filesname_list)
    return filesname_list

def dir_name(path):
    file_list = os.listdir(path)
    file_name_list = []
    for i in range(len(file_list)):
        file_name = path + '/' + file_list[i]
        # print(file_name)
        file_name_list.append(file_name)
    return file_name_list

def load_eod_data(dataset_name, data_path):
    '''
    :param data_path: stock data after propressed
    :return: train_data,train_label,text_data,text_label
    '''
    # get tickers name list
    tickers = files_name(dir_name(data_path))
    # print(tickers)
    for index,ticker in enumerate(tickers):
        single_eod_data = pd.read_csv(
            data_path + '/' + ticker + '.csv',header=None,delimiter=',').values
        # The start/end date of ACL18/KDD17 is different
        if dataset_name == 'ACL18':
            single_label = single_eod_data[148:, -2]
            single_eod_data = single_eod_data[148:,:-2]
            # print('single_eod_data.shape:{}'.format(single_eod_data.shape))
        elif dataset_name == 'KDD17':
            single_label = single_eod_data[29:, -2]
            single_eod_data = single_eod_data[29:, :-2]
            # print('single_eod_data.shape:{}'.format(single_eod_data.shape))
        # create total data and label
        if ticker == 'AAPL':
            eod_data = np.zeros(([len(tickers),single_eod_data.shape[0],single_eod_data.shape[1]])) # N,T,F
            labels_data = np.zeros([len(tickers),len(single_label)]) # N,T
        eod_data[index,:,:] = single_eod_data[:,:]
        labels_data[index,:] = single_label
    # divide dataset to training and testing 8:2 for ACL18 and 0-2266-2488 for KDD17
    data_len = eod_data.shape[1]
    # if dataset_name == 'ACL18':
    #     training_len = int(data_len * 0.8)
    # elif dataset_name == 'KDD17':
    #     training_len = 2266
    training_len = int(data_len * 0.6)
    test_len = int(data_len * 0.8)

    training_data,testing_data = eod_data[:,:training_len,:],eod_data[:,test_len:,:]

    training_labels,testing_labels = labels_data[:,:training_len],labels_data[:,test_len:]

    return training_data, training_labels, testing_data, testing_labels, tickers

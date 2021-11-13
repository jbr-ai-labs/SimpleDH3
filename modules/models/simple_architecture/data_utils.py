import pickle
import numpy as np
import torch
from enum import Enum

from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split
import embedding as emb

EMBEDDINGS_NAMES = ['light', 'normal', 'esm']

class Embedding(Enum):
    ONEHOT = 'onehot'
    # Protein sequence embedding
    PSE = 'pse'


def pad_data_collate(X, y, lengths, names, max_length=None):
    if max_length is not None:
        X = [x[:max_length] for x in X]
        y = [y[:max_length] for y in y]
        lengths = [np.minimum(l, max_length) for l in lengths]
    y_padded = rnn_utils.pad_sequence(y, batch_first=True)
    X_padded = rnn_utils.pad_sequence(X, batch_first=True, padding_value=20)
    # X_padded = rnn_utils.pad_sequence(X, batch_first=True)
    X, y, lengths, names = zip(*[[X_padded[i], y_padded[i], lengths[i], names[i]] for i in range(len(X_padded))])
    X = torch.stack(X)
    y = torch.stack(y)
    lengths = torch.tensor(lengths)
    # names = torch.tensor(names)
    return X, y, lengths, names


def pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test, max_length=None):
    if max_length is not None:
        X_train = [x[:max_length] for x in X_train]
        X_test = [x[:max_length] for x in X_test]
        y_train = [y[:max_length] for y in y_train]
        y_test = [y[:max_length] for y in y_test]
        lengths_train = [np.minimum(l, max_length) for l in lengths_train]
        lengths_test = [np.minimum(l, max_length) for l in lengths_test]
    y_train_padded = rnn_utils.pad_sequence(y_train, batch_first=True)
    X_train_padded = rnn_utils.pad_sequence(X_train, batch_first=True)
    train_dataset = [[X_train_padded[i], y_train_padded[i], lengths_train[i]] for i in range(len(X_train_padded))]

    X_test_padded = rnn_utils.pad_sequence(X_test, batch_first=True)
    y_test_padded = rnn_utils.pad_sequence(y_test, batch_first=True)
    test_dataset = [[X_test_padded[i], y_test_padded[i], lengths_test[i]] for i in range(len(X_test_padded))]

    return train_dataset, test_dataset


def get_dataset_angles(seq, angles):
    full_data = [[len(seq[x]), seq[x], angles[x]] for x in seq.keys()]
    test_data = full_data[:int(len(full_data) / 10)]
    train_data = full_data[int(len(full_data) / 10):]

    full_data.sort(key=lambda x: x[0], reverse=True)

    train_data.sort(key=lambda x: x[0], reverse=True)
    lengths_train = [train_data[i][0] for i in range(len(train_data))]
    X_train = [train_data[i][1] for i in range(len(train_data))]
    y_train = [train_data[i][2] for i in range(len(train_data))]

    test_data.sort(key=lambda x: x[0], reverse=True)
    lengths_test = [test_data[i][0] for i in range(len(test_data))]
    X_test = [test_data[i][1] for i in range(len(test_data))]
    y_test = [test_data[i][2] for i in range(len(test_data))]

    train_dataset, test_dataset = pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test)

    return train_dataset, test_dataset


def get_full_data_coordinates(seq, coordinates):
    return [[x, len(seq[x]), seq[x], coordinates[x]] for x in seq.keys()]


def dump_test_dataset(test_data, config):
    print(config['PATH_TO_TEST_DATASET_IDS'])
    with open(config['PATH_TO_TEST_DATASET_IDS'], 'w') as f:
        for data in test_data:
            #print('t ', data[0])
            f.write(str(data[0]) + '\n')


def get_unique_data_points(full_data):
    data_tensors = [x[2] for x in full_data]
    idx = set()
    for i in range(len(data_tensors)):
        for j in range(i + 1, len(data_tensors)):
            if data_tensors[i].shape[0] == data_tensors[j].shape[0] and torch.eq(data_tensors[i],
                                                                                 data_tensors[j]).all():
                idx.add(j)
    result = [x for i, x in enumerate(full_data) if i not in idx]
    return result


def cut_data_accoarding_to_length(full_data):
    low = 4
    hi = 25
    full_data = list(filter(lambda x: x[1] >= low and x[1] <= hi, full_data))
    return full_data


def get_test_dataset_coordinates(seq, coordinates):
    full_data = get_full_data_coordinates(seq, coordinates)
    data = [[x[1], x[2], x[3]] for x in full_data]
    return data


def get_dataset_coordinates(seq, coordinates, config, val_size=0.1, test_size=0.1, no_test = False):
    full_data = get_full_data_coordinates(seq, coordinates)
    full_data = get_unique_data_points(full_data)
    if no_test == False:
        train_data, test_data = train_test_split_len(full_data, test_size + val_size)
        test_data, val_data = train_test_split_len(test_data, val_size/(val_size+test_size))
    
        print("Train len: ", len(train_data)," Val len: ", len(val_data), " Test len: ", len(test_data))
    
        dump_test_dataset(test_data, config)
        print("Saved test ids into a file")
        train_data = [[x[1], x[2], x[3], x[0]] for x in train_data]
        val_data = [[x[1], x[2], x[3], x[0]] for x in val_data]
        test_data = [[x[1], x[2], x[3], x[0]] for x in test_data]
  
    elif no_test == True:
        train_data, val_data = train_test_split_len(full_data, val_size)
        print("Train len: ", len(train_data)," Val len: ", len(val_data))
        train_data = [[x[1], x[2], x[3], x[0]] for x in train_data]
        val_data = [[x[1], x[2], x[3], x[0]] for x in val_data]
        test_data = []
    return train_data, val_data, test_data 

def read_test_train_split(config):
    seq, coordinates = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME'],
                                         config['PATH_TO_COORD_EMBEDDED'], config['EMBEDDINGS_NAME'])
    ids = []
    with open(config['PATH_TO_TEST_DATASET_IDS'], 'r') as file:
        for line in file.readlines():
            ids.append(line[0:4])
    full_data = get_full_data_coordinates(seq, coordinates)
    full_data = get_unique_data_points(full_data)
    train_data = []
    test_data = []
    for data in full_data:
        if data[0] in ids:
            test_data.append([data[1], data[2], data[3], data[0]])
        else:
            train_data.append([data[1], data[2], data[3], data[0]])
    return train_data, test_data

def train_test_split_len(full_data, test_size=0.1):
    splited_by_len = {}
    train_data = []
    test_data = []
    counter = 0
    for x in full_data:
        lenth = len(x[2])
        if lenth in splited_by_len.keys():
            splited_by_len[lenth].append(x)
            counter += 1
        else:
            splited_by_len[lenth] = [x]
    
    for x in splited_by_len.keys():
        if len(splited_by_len[x]) > 2:
            train_data_len, test_data_len = train_test_split(splited_by_len[x], test_size=test_size, shuffle=True, random_state=0)
        else:
            train_data_len, test_data_len = splited_by_len[x], []
        train_data += train_data_len
        test_data += test_data_len
    #only for testing data splits
    #with open('../results/val_set', 'w') as f:
    #    for data in test_data:
    #        f.write(str(data[3]) + '\n')
    return train_data, test_data

def collate_f(batch):
    batch.sort(key=lambda x: x[0], reverse=True)
    lengths = [batch[i][0] for i in range(len(batch))]
    X = [batch[i][1] for i in range(len(batch))]
    y = [batch[i][2] for i in range(len(batch))]
    names = [batch[i][3] for i in range(len(batch))]
    #names = ['NAME' for i in range(len(batch))]
    data = pad_data_collate(X, y, lengths, names, max_length=50)
    return data


def get_test_dataloader(train_dataset, batch_size):
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_f)
    return test_dataloader


def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_f)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_f)

    return train_dataloader, val_dataloader


def get_sequence_data(seq_embedding_path, target_path, embedding_type):
    if embedding_type == EMBEDDINGS_NAMES[2]:
        # print('Seq emb path:', seq_embedding_path)
        seq = torch.load(seq_embedding_path)
    else:
        with open(seq_embedding_path, 'rb') as handle:
            seq = pickle.load(handle)
    with open(target_path, 'rb') as handle:
        target = pickle.load(handle)
    return seq, target


def get_embedded_data_angles(config):
    seq, angles = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME'], config["PATH_TO_ANGLES_EMBEDDED"], config["EMBEDDINGS_NAME"])
    return seq, angles


def get_embedded_test_data_coordinates(config, embedding=Embedding.PSE):
    if embedding == Embedding.PSE:
        seq_path = config["PATH_TO_TEST_SEQ_EMBEDDED"]
        coord_path = config["PATH_TO_COORD_EMBEDDED"]
    elif embedding == Embedding.ONEHOT:
        seq_path = config["PATH_TO_TEST_SEQ_ONEHOT"]
        coord_path = config["PATH_TO_COORD_EMBEDDED"]
    else:
        raise RuntimeError('Unknown embedding')
    seq, _ = get_sequence_data(seq_path, coord_path, config["EMBEDDINGS_NAME"])
    return seq


def get_embedded_data_coordinates(config, embedding=Embedding.PSE):
    if embedding == Embedding.PSE:
        seq_path = config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME']
        coord_path = config["PATH_TO_COORD_EMBEDDED"]
    elif embedding == Embedding.ONEHOT:
        seq_path = config["PATH_TO_SEQ_ONEHOT"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME']
        coord_path = config["PATH_TO_COORD_EMBEDDED"]
    else:
        raise RuntimeError('Unknown embedding')
    seq, coordinates = get_sequence_data(seq_path, coord_path, config["EMBEDDINGS_NAME"])
    return seq, coordinates

def train_val_split(config):
    seq, coordinates = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME'],
                                         config['PATH_TO_COORD_EMBEDDED'], config['EMBEDDINGS_NAME'])
    
    ids_test = []
    with open(config['PATH_TO_TEST_DATASET_IDS'], 'r') as file:
        for line in file.readlines():
            ids_test.append(line[0:4])
    ids_val = []
    with open('../results/val_set', 'r') as file:
        for line in file.readlines():
            ids_val.append(line[0:4])
    
    full_data = get_full_data_coordinates(seq, coordinates)
    full_data = get_unique_data_points(full_data)
    train_data = []
    val_data = []
    test_data = []
    for data in full_data:
        if data[0] in ids_test:
            test_data.append([data[1], data[2], data[3], data[0]])
        elif data[0] in ids_val:
            val_data.append([data[1], data[2], data[3], data[0]])
        else:
            train_data.append([data[1], data[2], data[3], data[0]])
    return train_data, val_data, test_data

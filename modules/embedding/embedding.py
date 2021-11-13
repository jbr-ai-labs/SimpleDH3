import sys
import numpy as np

import torch
import torch.nn as nn

from src.models.alphabets import Uniprot21


def load_data(path_to_data):
    alphabet = Uniprot21()

    dataset = {}
    with open(path_to_data, 'rb') as f:
        for line in f:
            name, seq = line.split()
            seq = alphabet.encode(seq)
            dataset[name] = seq

    return dataset


def unstack_lstm(lstm):
    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)

            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
        layers.append(layer)
        in_size = 2 * hidden_dim
    return layers


def featurize_dict(dataset, lm_embed, lstm_stack, proj, use_cuda=False):
    embedding = {}
    for name, seq in dataset.items():
        with torch.no_grad():
            seq = torch.from_numpy(seq).long().unsqueeze(0)
            if use_cuda:
                seq = seq.cuda()
            emb = featurize(seq, lm_embed, lstm_stack, proj)
            emb = emb.squeeze(0).cpu()
            embedding[name] = emb
    return embedding


def featurize(x, lm_embed, lstm_stack, proj):
    h = lm_embed(x)
    for lstm in lstm_stack:
        h, _ = lstm(h)
    h = proj(h.squeeze(0)).unsqueeze(0)
    return h


def get_embeddings(path_to_model, path_to_data='antibodies_data/test_seq', device=-1):
    dataset = load_data(path_to_data)

    d = device
    use_cuda = (d != -1) and torch.cuda.is_available()

    if d >= 0:
        torch.cuda.set_device(d)

    encoder = torch.load(path_to_model)
    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    if use_cuda:
        lm_embed.cuda()
        for lstm in lstm_stack:
            lstm.cuda()
        proj.cuda()

    ## featurize the sequences
    print('# featurizing data', file=sys.stderr)
    dict = featurize_dict(dataset, lm_embed, lstm_stack, proj, use_cuda=use_cuda)

    del lm_embed
    del lstm_stack
    del proj
    del encoder

    return dict


def get_angles(path_to_data):
    dataset_angles = {}
    with open(path_to_data, 'rb') as f:
        head = True
        for line in f:
            if head:
                name, l = line.split()
                head = False
                dataset_angles[name] = []
            elif line != b'\n':
                aa, phi, psi = line.split()
                phi = float(phi)
                psi = float(psi)
                dataset_angles[name].append([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)])
            else:
                head = True
                dataset_angles[name] = torch.FloatTensor(dataset_angles[name])
    return dataset_angles


def get_coordinates(path_to_data):
    dataset_coord = {}
    with open(path_to_data, 'r') as f:
        line = f.readline()
        while line:
            name, l = line.split()
            dataset_coord[name] = []
            aa_coord = []
            for i in range(int(l)):
                for x in map(float, f.readline().split()):
                    aa_coord.append(x)
                if i % 3 == 2:
                    dataset_coord[name].append(aa_coord)
                    aa_coord = []
            dataset_coord[name] = torch.FloatTensor(dataset_coord[name])
            f.readline()
            line = f.readline()
    return dataset_coord


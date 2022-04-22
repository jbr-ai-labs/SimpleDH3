from collections import OrderedDict

import torch
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

import sys
sys.path.append("")

from models.simple_architecture.model import SimpleCharRNNUnit, SimpleCharRNN, PSEModel
from config_loader import load_config
from models.simple_architecture.simplemodel_coordinates.train import ComplexLoss
from metrics.metric import coordinate_metrics, get_dist_metrics
from models.simple_architecture.data_utils import (collate_f,
                                                   get_full_data_coordinates,
                                                   get_sequence_data,
                                                   get_unique_data_points)
from models.simple_architecture.model import rescale_prediction, rescale_prediction_rotation
import numpy as np
from sklearn.model_selection import KFold

MODELS_NAMES = ['simple', 'simple_coordinates']
CORRECTORS_NAMES = ['distances', 'rotate_normal']

string_to_corrector_dict = {
    CORRECTORS_NAMES[0]: rescale_prediction,
    CORRECTORS_NAMES[1]: rescale_prediction_rotation
}

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def string_to_corrector(s):
    if s not in string_to_corrector_dict:
        raise ValueError('Not a valid corrector string')
    return string_to_corrector_dict[s]

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='simple_coordinates',
                        choices=MODELS_NAMES,
                        help='Which model to run')
    parser.add_argument('--saved_model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--input_size', type=int, required=True,
                        help='Model input size')
    parser.add_argument('--output_size', type=int, required=True,
                        help='Model output size')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='Model hidden dim')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Model layers number')
    parser.add_argument('--data_path_seq', type=str, required=True,
                        help='Path to data file with embedded sequences')
    parser.add_argument('--data_path_coord', type=str, required=True,
                        help='Path to data file with target coordinates')
    parser.add_argument('--test_data_ids_path', type=str, required=True,
                        help='Path to test data file ids of pdb')
    parser.add_argument('--use_corrector', type=boolean_string, required=False,
                        help='Use corrector for coordinates?')
    parser.add_argument('--corrector', type=str, default='distances',
                        choices=CORRECTORS_NAMES,
                        help='Which corrector to use')
    parser.add_argument('--data_path_gdt_ts', type=str, required=True, help='Path to global distance test')
    parser.add_argument('--data_path_tm_score', type=str, required=True, help='Path to TM score')
    parser.add_argument('--data_path_rmsd', type=str, required=True, help='Path to RMSD scores')
    return parser


BATCH_SIZE = 500


def load_test_data_to_dataloader(args):
    config = load_config()
    seq, coordinates = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_test_' + config['EMBEDDINGS_NAME'], config['PATH_TO_COORD_EMBEDDED'] + 'test', config['EMBEDDINGS_NAME'])
    ids = []
    with open(args['test_data_ids_path'], 'r') as file:
        for line in file.readlines():
            ids.append(line[0:4])
    seq = {id: seq[id] for id in ids}
    coordinates = {id: coordinates[id] for id in ids}
    test_data = [[x[1], x[2], x[3], x[0]] for x in get_full_data_coordinates(seq, coordinates)]
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_f)
    return test_dataloader

def load_test_k_data_to_dataloader(k_folds):
    config = load_config()
    seq, coordinates = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_test_' + config['EMBEDDINGS_NAME'],
                                         config['PATH_TO_COORD_EMBEDDED'] + 'test', config['EMBEDDINGS_NAME'])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    full_data = get_full_data_coordinates(seq, coordinates)
    full_data = get_unique_data_points(full_data)
    return kfold.split(full_data)

def prepare_padded_tensors(preds, targets, lengths):
    prepared = []
    for pred, target, length in zip(torch.unbind(preds), torch.unbind(targets), lengths):
        prepared.append((pred[:length, :], target[:length, :]))
    return prepared

def calc_atom_dist(coord1, coord2):
    squared_dist = np.sum((np.array(coord1) - np.array(coord2)) ** 2, axis=0)
    return np.sqrt(squared_dist)

def predict_test(model, model_no_corrector, test_data):
    loss = ComplexLoss(on_cpu=True)
    tensors_to_save = {}
    with torch.no_grad():
        for inputs, targets, lengths, names in test_data:
            preds, lengths, hiddens = model(inputs, lengths)
            # preds_, lengths_, hiddens_ = model_no_corrector(inputs, lengths)
            loss_value = loss(preds, targets, lengths).item()
            print(f'Loss value: {loss_value}')
            metrics = coordinate_metrics(preds, targets, lengths, on_cpu=True)
            dist_metrics = get_dist_metrics(preds, targets)
            # metrics_ = coordinate_metrics(preds_, targets, lengths_, on_cpu=True)
            print(metrics)
            print(dist_metrics)
            print('Structure with RMSD > 5:')
            min_rmsd = 100500
            min_name = ''
            index = 0
            for ind in range(len(names)):
                if metrics['rmsd_batch'][ind] > 5:
                    print(names[ind], ': ', metrics['rmsd_batch'][ind])
                if metrics['rmsd_batch'][ind] < min_rmsd:
                    min_rmsd = metrics['rmsd_batch'][ind]
                    min_name = names[ind]
            print(lengths.numpy())
            # for ind in range(len(names)):
            #     print(lengths[ind].numpy(), ', ')
            # for ind in range(len(names)):
            #     print(metrics['rmsd_batch'][ind].numpy(), ',')
            # print('The best prediction:\n')
            print(min_name, min_rmsd)
            # bug if test data bigger batch size
            for name, item in zip(names, prepare_padded_tensors(preds, targets, lengths)):
                tensors_to_save[name] = item
    return tensors_to_save

def get_gdt_ts(pred, target):
    number = 0
    p = np.zeros(3)
    for aa_p, aa_t in zip(pred, target):
        dist_aa = torch.dist(aa_p[3:6], aa_t[3:6], p=2) # distance between C_alpha
        number += 1
        if dist_aa <= 1:
            p[0] += 1
        if dist_aa <= 2:
            p[1] += 1
        if dist_aa <= 4:
            p[2] += 1
    p /= number
    return (np.sum(p) / 3 * 100)

def get_tm_score(pred, target):
    pred = pred.view(-1, 3)
    target = target.view(-1, 3)
    l = len(pred)
    S = 0
    normalizer = 1.24 * (l - 15) ** (1 / 3) - 1.8
    for i in range(l):
        S += 1 / (1 + ((pred[i] - target[i]).norm() / normalizer) ** 2)
    return S / l
def get_rmsd(pred, target):
    pred = pred.view(-1, 3)
    target = target.view(-1, 3)
    msd = torch.sum((pred - target) ** 2, dim=-1).sum(dim=-1) / len(pred)
    return torch.sqrt(msd)

def save_tensors(tensors_to_save, path_to_gdt_ts, path_to_tm_score, path_to_rmsd, path=None):
    test_result_folder = ''
    test_result_folder += config['PATH_TO_TEST_RESULTS'] if path is None else path
    Path(test_result_folder).mkdir(parents=True, exist_ok=True)
    gdt_ts = list()
    tm_score = list()
    rmsd_score = list()
    for i, pt in tensors_to_save.items():
        torch.save(pt[0], test_result_folder + f'{i}_pred.pt')
        torch.save(pt[1], test_result_folder + f'{i}_target.pt')
        gdt_ts.append((i, get_gdt_ts(pt[0], pt[1])))
        tm_score.append((i, get_tm_score(pt[0], pt[1])))
        rmsd_score.append((i, get_rmsd(pt[0], pt[1])))
    with open(path_to_gdt_ts, 'w') as f:
        for gdt in gdt_ts:
           print(f"{gdt[0]} {gdt[1]}", file=f)
    with open(path_to_tm_score, 'w') as f:
        for tm in tm_score:
           print(f"{tm[0]} {tm[1]}", file=f)
    with open(path_to_rmsd, 'w') as f:
        for r in rmsd_score:
           print(f"{r[0]} {r[1]}", file=f)

if __name__ == '__main__':
    parser = get_parser()
    config = load_config()
    args = vars(parser.parse_known_args()[0])
    model_arg = args['model']
    model = None
    corrector = None

    if model_arg == 'simple_coordinates':
        input_size = args['input_size']
        output_size = args['output_size']
        hidden_dim = args['hidden_dim']
        n_layers = args['n_layers']
        use_corrector = args['use_corrector']
        corrector = string_to_corrector(args['corrector'])
        device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SimpleCharRNN(input_size, output_size, hidden_dim, n_layers, device,
                              bilstm=True, corrector=corrector)
        model_no_corrector = SimpleCharRNN(input_size, output_size, hidden_dim, n_layers, device,
                              bilstm=True)
        saved_model_path = args['saved_model_path']
        state_dict = torch.load(saved_model_path, map_location=lambda storage, loc: storage)
        if isinstance(state_dict, OrderedDict):
            model.load_state_dict(state_dict)
            model_no_corrector.load_state_dict(state_dict)
        else:
            model = state_dict
            model_no_corrector = state_dict
        model.eval()
        model.use_corrector(use_corrector)
        model.cpu()
        model.device = torch.device('cpu')
        model_no_corrector.eval()
        model_no_corrector.use_corrector(use_corrector)
        model_no_corrector.cpu()
        model_no_corrector.device = torch.device('cpu')
        test_dataloader = load_test_data_to_dataloader(args)

        tensors_to_save = predict_test(model, model_no_corrector, test_dataloader)
        save_tensors(tensors_to_save, args['data_path_gdt_ts'], args['data_path_tm_score'], args['data_path_rmsd'])
    else:
        raise NotImplemented('Support only for simple model predicting coordinates.')


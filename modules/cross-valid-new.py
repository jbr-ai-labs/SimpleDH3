from collections import OrderedDict

import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
from main import *
from metrics.metric import coordinate_metrics
from models.simple_architecture.model import SimpleCharRNN
from models.simple_architecture.simplemodel_coordinates.train import ComplexLoss
from predict_test_data import CORRECTORS_NAMES, boolean_string, save_tensors, string_to_corrector, \
    load_test_data_to_dataloader, load_test_k_data_to_dataloader, prepare_padded_tensors, string_to_corrector_dict
from models.simple_architecture.data_utils import (collate_f,
                                                   get_full_data_coordinates,
                                                   get_sequence_data,
                                                   get_unique_data_points)
config = load_config()

def extend_parser(parser):
    parser.add_argument('--iterations_cv', type=int, default=8,
                        help='Number of cross-validation iterations')
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
    return parser

def evaluate_test_metircs(model, test_data, iter):
    loss = ComplexLoss(on_cpu=True)
    with torch.no_grad():
        for inputs, targets, lengths, names in test_data:
            preds, lengths, hiddens = model(inputs, lengths)
            loss_value = loss(preds, targets, lengths).item()
            print(f'Loss value: {loss_value}')
            metrics = coordinate_metrics(preds, targets, lengths, on_cpu=True)
            print(metrics)
            with open("fold_res" + iter + ".txt", "w") as myfile:
                for ind in range(len(names)):
                    myfile.write(names[ind] + ' : ' + str(metrics['rmsd_batch'][ind].item()) + '\n')
            # bug if test data bigger batch size

    return loss_value, metrics

def load_model(model, saved_model_path):
    state_dict = torch.load(saved_model_path, map_location=lambda storage, loc: storage)
    if isinstance(state_dict, OrderedDict):
        model.load_state_dict(state_dict)
    else:
        model = state_dict
    model.eval()
    model.use_corrector(use_corrector)
    model.cpu()
    model.device = torch.device('cpu')

    return model

def process_results(losses, metrics):
    print('Cross-validation results:')
    print('Mean loss:', np.mean(losses))
    for metric in metrics[0].keys():
        metric_l = []
        for iteration in metrics:
            metric_l.append(iteration[metric])
        print(metric + ':', np.mean(metric_l))


if __name__ == '__main__':
    main_logger = get_logger()
    main_parser = get_parser()
    main_parser = extend_parser(main_parser)
    args = vars(main_parser.parse_known_args()[0])
    use_backup = args['use_backup']
    choice_model = args['model']
    debug = args['debug']
    embedding = args['embedding']
    model_save_prefix = args['save_prefix']
    args['cross-valid'] = True
    test_ids_path = args['test_data_ids_path']
    seq, coordinates = get_sequence_data(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_' + config['EMBEDDINGS_NAME'],
                                         config['PATH_TO_COORD_EMBEDDED'], config['EMBEDDINGS_NAME'])
    # print('Len seq data cross-val:', len(seq))
    full_data = get_full_data_coordinates(seq, coordinates)
    print('full:', len(full_data))
    full_data = get_unique_data_points(full_data)
    print('full unique:', len(full_data))
    test_dataloader_k = load_test_k_data_to_dataloader(args['iterations_cv'])
    for i, (train_ids, test_ids) in enumerate(test_dataloader_k):
        print("Iteration:", i + 1)
        print(len(test_ids))
        test_names = []
        for j in range(4000):
            if j >= len(full_data):
                break
            if j in test_ids:
                test_names.append(full_data[j][0])
        print('Test names: ', test_names)
        
        with open(args['test_data_ids_path'], 'w') as file:
            for data in test_names:
                file.write(data + '\n')
        print('Saved test ids for cross-valid')
        try:
            print("Start train")
            args['save_prefix'] = model_save_prefix + str(i) 
            simplemodel_coord_train(main_logger, args, debug=debug, embedding=embedding)
            print("Finish train")
        except:
            if debug:
                raise
            main_logger.exception('Exception while training a model')
            exit(1)

        input_size = args['model_input_size']
        output_size = args['model_output_size']
        hidden_dim = args['model_hidden_dim']
        n_layers = args['n_layers']
        device = 'cpu'  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SimpleCharRNN(input_size, output_size, hidden_dim, n_layers, device,
                              bilstm=True, corrector=None)

        losses = []
        metrics = []
        use_corrector = False
        corrector = None
        saved_model_path = '../results/' + model_save_prefix + str(i) + '.sav'
        model = load_model(model, saved_model_path)
        model.corrector = corrector
        model.use_corrector(use_corrector)
        test_dataloader = load_test_data_to_dataloader(args)

        loss, metrs = evaluate_test_metircs(model, test_dataloader, str(i))
        losses.append(loss)
        with open("metrs" + str(i) + ".txt", "w") as myfile:
            myfile.write(','.join(str(x) for x in metrs['rmsd_batch'].tolist()))
        plt.clf()
        plt.xlabel("RMSD")
        plt.ylabel("")
        plt.scatter(metrs['rmsd_batch'].tolist(), [1 for _ in range(len(metrs['rmsd_batch'].tolist()))], label='Train loss')
        plt.tight_layout()
        plt.autoscale()
        plt.savefig('rmsd_plot' + str(i) + '.png', figsize=(19.20, 10.80), bbox_inches="tight")
        del metrs['rmsd_batch']
        metrics.append(metrs)

        shutil.copy(args['test_data_ids_path'], args['test_data_ids_path']+str(i)) 

    process_results(losses, metrics)

    best_id = np.argmax(losses)
    print('Best model id: ', best_id)
    saved_model_path = '../results/' + model_save_prefix + str(best_id) + '.sav'
    model = load_model(model, saved_model_path)

    model_path = '../results/' + model_save_prefix + '.sav'
    torch.save(model.state_dict(), model_path)

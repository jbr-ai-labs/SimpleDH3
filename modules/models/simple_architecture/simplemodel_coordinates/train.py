import os
import os.path
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb
import tensorflow as tf

import models.simple_architecture.data_utils as data_utils
import models.simple_architecture.train_utils as train_utils
from config_loader import load_config
from models.simple_architecture.model import SimpleCharRNNUnit, SimpleCNN, SimpleCharRNN, rescale_prediction
from models.simple_architecture.data_utils import Embedding
from metrics.metric import distance_between_atoms, angles_between_atoms, torsion_angles_between_atoms

MODEL_NAME = 'simple-coordinates'

class DistanceLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.on_cpu = on_cpu

    def reshape_loop(self, loop):
        return loop.reshape(loop.shape[0], -1, 3)

    def forward(self, pred, target):
        pred = self.reshape_loop(pred)
        target = self.reshape_loop(target)
        distances_pred = distance_between_atoms(pred)
        distances_target = distance_between_atoms(target)
        z = self.mse_loss_function(distances_pred, distances_target)
        return z


class AnglesLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.on_cpu = on_cpu

    def forward(self, pred, target, lengths):
        angles_pred = angles_between_atoms(pred, lengths, self.on_cpu)
        angles_target = angles_between_atoms(target, lengths, self.on_cpu)
        z = self.mse_loss_function(angles_pred, angles_target)
        return z

class TorsionAnglesLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.on_cpu = on_cpu

    def forward(self, pred, target):
        torsion_pred = torsion_angles_between_atoms(pred, self.on_cpu)
        torsion_target = torsion_angles_between_atoms(target, self.on_cpu)
        z = self.mse_loss_function(torsion_pred, torsion_target)
        return z

class ComplexLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.distance_loss_function = DistanceLoss(on_cpu)
        self.angles_loss_function = AnglesLoss(on_cpu)
        self.torsion_loss_function = TorsionAnglesLoss(on_cpu)

    def forward(self, pred, target, lengths):
        mse_loss = self.mse_loss_function(pred, target)
        distance_loss = self.distance_loss_function(pred, target)
        angles_loss = self.angles_loss_function(pred, target, lengths)
        torsion_loss = self.torsion_loss_function(pred, target)
        z = mse_loss = mse_loss + distance_loss + torsion_loss + angles_loss
        with open('../results/coord_loss', 'a') as f:
            f.write(str(mse_loss.item()) + '\n')
        with open('../results/distance_loss', 'a') as f:
            f.write(str(distance_loss.item()) + '\n')
        with open('../results/angles_loss', 'a') as f:
            f.write(str(angles_loss.item()) + '\n')
        with open('../results/torsion_loss', 'a') as f:
            f.write(str(torsion_loss.item()) + '\n')
        #z = mse_loss + angles_loss + distance_loss
        # z = mse_loss + distance_loss  # + angles_loss
        # todo add distance between ends loss
        return z


def parse_parameters(args):
    # todo add default params or throw exception
    model_input_size = args['model_input_size']
    model_output_size = args['model_output_size']
    model_hidden_dim = args['model_hidden_dim']
    learning_rate = args['learning_rate']
    n_layers = args['n_layers']
    batch_size = args['batch_size']
    epochs = args['epochs']
    test_size = args['test_size']
    return {
        "input_size": model_input_size,
        "output_size": model_output_size,
        "hidden_dim": model_hidden_dim,
        "learning_rate": learning_rate,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "epochs": epochs,
        "test_size": test_size
    }


def simplemodel_coord_train(logger, args, use_backup=False, debug=False, embedding=Embedding.PSE):
    params = parse_parameters(args)
    save_prefix = args['save_prefix']
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, coord = data_utils.get_embedded_data_coordinates(config, embedding=embedding)
    print('Len sequences data:', len(seq))
    print('Len coordinates data:', len(coord))
    # seq_test = data_utils.get_embedded_test_data_coordinates(config, embedding=embedding)
    # test_data = data_utils.get_test_dataset_coordinates(seq_test, coord)
    
    # splitting into train/val/test, test set is saved into a file
    if args['cross_valid'] ==  False:
        train_data, val_data, test_data = data_utils.get_dataset_coordinates(seq, coord, config, val_size=0.1, test_size=0.1, no_test = args['no_test'])
        #train_data, val_data, test_data = data_utils.train_val_split(config)  #testing with cross-val sets

    elif args['cross_valid'] == True:
        print('Cross valid is true')
        # in cross-validation test set is read from the file
        train_data, test_data = data_utils.read_test_train_split(config)
        train_data, val_data = data_utils.train_test_split_len(train_data, test_size = 0.11)
        
    
    #test_dataloader = data_utils.get_test_dataloader(test_data, params['batch_size'])
    test_dataloader = None
    train_dataloader, val_dataloader = data_utils.get_dataloaders(train_data, val_data, params['batch_size'])

    # model = SimpleCNN(params['input_size'], params['output_size'], hidden_dim=128, n_layers=3, device=device,
    #                   kernel_size=3)
    # model = SimpleRNNSphere(params['input_size'], 12, params['hidden_dim'], params['n_layers'],
    #                         device)
    # model = SimpleCNN(params['input_size'], params['output_size'], params['hidden_dim'], params['n_layers'],
    #                   kernel_size=3, device=device)
    model = SimpleCharRNN(params['input_size'], params['output_size'], params['hidden_dim'], params['n_layers'], device,
                          bilstm=True)
    
    start_epoch, model = train_utils.try_load_model_backup(model, MODEL_NAME, use_backup, logger, config)
    model.to(device)
    # model.use_corrector(True)

    #if not debug:
        #train_utils.initialize_wandb(model, config, params['n_layers'], params['batch_size'],
                                     #'simple-model-coordinates', params['hidden_dim'])
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfb_path_train = 'tensorboard/' + current_time + '/train_epoch'
    tfb_path_val = 'tensorboard/' + current_time + '/valid_epoch'

    train_summary_writer = tf.summary.create_file_writer(tfb_path_train, name="Train")
    valid_summary_writer = tf.summary.create_file_writer(tfb_path_val, name="Validate")

    loss = ComplexLoss(on_cpu=debug)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=100,
    #                                               cycle_momentum=False)

    Path(config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"]).mkdir(parents=True, exist_ok=True)

    model = train_utils.train_model(train_dataloader, val_dataloader, test_dataloader, model, MODEL_NAME, loss, optimizer,
                            params['epochs'],
                            train_summary_writer, valid_summary_writer,
                            device,
                            config,
                            train_utils.coordinates_metrics_logger,
                            params=params,
                            scheduler=scheduler,
                            start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                            num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"],
                            debug=debug,
                            early_stopping_steps=args['early_stopping_steps'], early_stopping_gap=args['early_stopping_gap'],
                            save_all=args['save_all']
                            )
    #train_utils.write_training_epoch(config, 0, MODEL_NAME, logger)
    if save_prefix is not None:
        model_path = '../results/' + save_prefix + '.sav'
        torch.save(model.state_dict(), model_path)
        print('Model saved with name: ', save_prefix,  '.sav')

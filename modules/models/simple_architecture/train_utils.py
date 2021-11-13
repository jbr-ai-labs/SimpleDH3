import os
import copy
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.utils.rnn as rnn_utils
import tensorflow as tf
import wandb
from tqdm import tqdm
from matplotlib import rcParams
from metrics.metric import coordinate_metrics, rmsd

def get_rmsd(preds, targets, lengths):
    preds = preds.reshape(preds.shape[0], -1, 3)
    targets = targets.reshape(preds.shape[0], -1, 3)
    rmsd_batch = rmsd(preds, targets, lengths)
    return rmsd_batch.sum()


def get_mae(preds, targets, lengths):
    preds = preds.reshape(preds.shape[0], -1, 3)
    targets = targets.reshape(preds.shape[0], -1, 3)
    mae_batch = (preds - targets).norm(dim=-1).mean(-1)
    return mae_batch.sum()


def train_model(train_dataloader, val_dataloader, test_dataloader, model, model_name, loss, optimizer, num_epochs,
                logger_train, logger_val,
                device,
                config,
                metrics_logger,
                scheduler=None,
                model_backup_path=None, start_epoch=0, num_epoch_before_backup=100, debug=False,
                early_stopping_steps=None, early_stopping_gap=0.01,
                save_all=False, params={}):

    start = time.time()
    train_iter = 0
    val_iter = 0
    epoch_rmsd_no_improvement = 0
    epoches_witout_improvement = 0
    best_val_score = torch.tensor(999999.0)
    best_rmsd_score = torch.tensor(999999.0)
    best_model = copy.deepcopy(model)
    losses_train = {}
    losses_validate = {}
    for epoch in range(start_epoch, num_epochs):
        #logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        if early_stopping_steps is not None and epoches_witout_improvement > early_stopping_steps:
            print('Early stop LOSS at epoch ', epoch)
            break
        if early_stopping_steps is not None and  epoch_rmsd_no_improvement > early_stopping_steps:
            print('Early stop RMSD at epoch ', epoch)
            break
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                dataloader = train_dataloader
                if scheduler is not None:
                    scheduler.step()
                model.train()
            elif phase == 'val':
                dataloader = val_dataloader
                model.eval()
            else:
                if epoch % 10 != 0 or test_dataloader is None:
                    continue
                dataloader = test_dataloader
                model.eval()

            running_loss = 0.

            rmsd = 0
            mae = 0
            num_points = len(dataloader.dataset)

            for inputs, targets, lengths, names in tqdm(dataloader):
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                # print(lengths.shape)
        
                targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
                # print(targets)
                targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
                targets = targets.to(device)
                # print(targets.shape)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds, lengths, hiddens = model(inputs, lengths)
                    loss_value = loss(preds, targets, lengths)

                    rmsd += get_rmsd(preds, targets, lengths) / num_points
                    mae += get_mae(preds, targets, lengths) / num_points
                    
                    if phase == 'train':

                        loss_value.backward()
                        optimizer.step()
                        last_train_targets = targets
                        train_iter += 1
                        # if epoch % 10 == 0:
                        if debug:
                            metrics_logger(preds, targets, lengths, logger_train, on_cpu=True, phase=phase)
                        else:
                            metrics_logger(preds, targets, lengths, logger_train,
                                           on_cpu=False, phase=phase, step=train_iter)
                    elif phase == 'val':
                        last_val_targets = targets
                        val_iter += 1
                        # if epoch % 10 == 0:
                        if debug:
                            metrics_logger(preds, targets, lengths, logger_val, on_cpu=True, phase=phase)
                        else:
                            metrics_logger(preds, targets, lengths, logger_val,
                                           on_cpu=False, phase=phase, step=val_iter)
                    else:
                        if debug:
                            pass #metrics_logger(preds, targets, lengths, logger, on_cpu=True, phase=phase)
                        else:
                            pass #metrics_logger(preds, targets, lengths, wandb, on_cpu=False, phase=phase)

                # statistics
                running_loss += loss_value.item()

            if not debug:
                if phase == 'train':
                    print(rmsd)
                    with logger_train.as_default():
                        tf.summary.scalar("MAE", mae.item(), step=epoch)
                        tf.summary.scalar("RMSD", rmsd.item(), step=epoch)
                elif phase == 'val':
                    with logger_val.as_default():
                        tf.summary.scalar("MAE", mae.item(), step=epoch)
                        tf.summary.scalar("RMSD", rmsd.item(), step=epoch)
                else:
                    pass

            epoch_loss = running_loss / len(dataloader)

            #logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                if not debug:
                    with logger_train.as_default():
                        tf.summary.scalar('Loss', epoch_loss, step=epoch)
                losses_train[epoch] = epoch_loss
                print(epoch, epoch_loss)
            elif phase == 'val':
                if not debug:
                    with logger_val.as_default():
                        tf.summary.scalar('Loss', epoch_loss, step=epoch)
                epoches_witout_improvement += 1
                epoch_rmsd_no_improvement += 1
                if (best_val_score - epoch_loss) >= early_stopping_gap:
                    best_val_score = epoch_loss
                    epoches_witout_improvement = 0
                    best_model = copy.deepcopy(model)
                if (best_rmsd_score - rmsd) >= early_stopping_gap:
                    best_rmsd_score = rmsd
                    epoch_rmsd_no_improvement = 0
                    best_model = copy.deepcopy(model)
                losses_validate[epoch] = epoch_loss
                print(epoch, epoch_loss)
            else:
                if not debug:
                    #wandb.log({"Test loss": epoch_loss})
                    pass

        if epoch % num_epoch_before_backup == 0 and model_backup_path:
            torch.save(best_model.state_dict(), model_backup_path + 'simplemodel_coord_backup')
            #write_training_epoch(config, epoch, model_name, logger)
        if save_all:
            torch.save(best_model.state_dict(), model_backup_path + 'simplemodel_coord_backup_e' + str(epoch))

    # with torch.no_grad():
    #     for inputs, targets, lengths, names in tqdm(val_dataloader):
    #         inputs = inputs.to(device)
    #         lengths = lengths.to(device)
    #         targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
    #         targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
    #         targets = targets.to(device)
    #         preds, lengths, hiddens = best_model(inputs, lengths)
    #         metrics = coordinate_metrics(preds, targets, lengths, on_cpu=False)
    #         with open("search.txt", "a") as myfile:
    #             myfile.write(str(params))
    #             myfile.write(str(metrics))
    #             myfile.write("\n")
    end = time.time()
    ("-----Time spend for training: ", end-start)
    draw_plot(losses_train, losses_validate)
    return best_model


def draw_plot(losses_train, losses_validate):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(list(losses_train.keys()), list(losses_train.values()), label='Train loss')
    plt.plot(list(losses_validate.keys()), list(losses_validate.values()), label='Test loss')
    plt.tight_layout()
    rcParams.update({'figure.autolayout': True})
    plt.autoscale()
    plt.savefig('../results/losses_plot.png', figsize=(19.20, 10.80), bbox_inches="tight")


def write_training_epoch(config, epoch, model, logger):
    if model == 'simple-coordinates':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"
    else:
        logger.error(f'Error: no such model {model}')
        return
    with open(config[config_path_name], 'w') as f:
        f.write(f"{epoch}")


def check_training_epoch(config, model, logger):
    if model == 'simple-coordinates':
        config_path_name = "PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"
    else:
        logger.error(f'Error: no such model {model}')
        return
    if not os.path.isfile(config[config_path_name]):
        return 0
    with open(config[config_path_name], 'r') as f:
        epoch = int(f.read())
        return epoch


def try_load_unfinished_model(logger, config, model):
    if model == 'simple-coordinates':
        config_path_name = "PATH_TO_SIMPLEMODEL_COORD_BACKUP"
    else:
        logger.error(f'Error: no such model {model}')
        return
    try:
        state = torch.load(config[config_path_name])
        return state
    except:
        logger.exception(f'Error loading unfinished {model}')


def coordinates_metrics_logger(preds, targets, lengths, logger, on_cpu=False, phase='val', step=None):
    metrics = coordinate_metrics(preds, targets, lengths, on_cpu)
    train_tag = ''
    if phase == 'train':
        train_tag = ' train'
    elif phase == 'test':
        train_tag = ' test'

    if on_cpu:
        logger.info(f"MAE batch{train_tag}: {metrics['mae']}")
        logger.info(f"Min MAE batch{train_tag}: {metrics['mae_min']}")
        logger.info(f"Max MAE batch{train_tag}: {metrics['mae_max']}")
        logger.info(f"Median MAE batch{train_tag}: {metrics['mae_median']}")
        logger.info(f"RMSD batch{train_tag}: {metrics['rmsd']}")
        logger.info(f"Min RMSD batch{train_tag}: {metrics['rmsd_min']}")
        logger.info(f"Max RMSD batch{train_tag}: {metrics['rmsd_max']}")
        logger.info(f"Median RMSD batch{train_tag}: {metrics['rmsd_median']}")
        logger.info(f"Distance deviation between neighbours{train_tag}: {metrics['diff_neighbours_dist']}")
        logger.info(f"Angles deviation{train_tag}: {metrics['diff_angles']}")
        #logger.info(f"Torsion angles deviation{train_tag}: {metrics['diff_torsion_angles']}")
    else:
        with logger.as_default():
            tf.summary.scalar("MAE batch", metrics['mae'].item(), step=step)
            tf.summary.scalar("Min MAE batch", metrics['mae_min'].item(), step=step)
            tf.summary.scalar("Max MAE batch", metrics['mae_max'].item(), step=step)
            tf.summary.scalar("Median MAE batch", metrics['mae_median'].item(), step=step)
            tf.summary.scalar("RMSD batch", metrics['rmsd'].item(), step=step)
            tf.summary.scalar("Min RMSD batch", metrics['rmsd_min'].item(), step=step)
            tf.summary.scalar("Max RMSD batch", metrics['rmsd_max'].item(), step=step)
            tf.summary.scalar("Median RMSD batch", metrics['rmsd_median'].item(), step=step)
            tf.summary.scalar("Distance deviation between neighbours batch", metrics['diff_neighbours_dist'].item(), step=step)
            tf.summary.scalar("Angles deviation batch", metrics['diff_angles'].item(), step=step)
            #tf.summary.scalar("Torsion angles deviation", metrics['diff_torsion_angles'].item(), step=step)

def try_load_model_backup(model, model_name, use_backup, logger, config):
    start_epoch = check_training_epoch(config, model_name, logger) if use_backup else 0
    logger.info(f'Starting training from epoch {start_epoch}')
    if start_epoch > 0:
        state = try_load_unfinished_model(logger, config, model_name) if use_backup else None
        if state:
            logger.info(f'Successfully loaded backup model')
            model.load_state_dict(state)
    return start_epoch, model


def initialize_wandb(model, config, n_layers, batch_size, model_name, hidden_dim):
    wandb.init(project=config["PROJECT_NAME"],
               name=f"{model_name} n_layers={n_layers} batch_size={batch_size} hidden_dim={hidden_dim} with_scheduler")
    wandb.watch(model)

import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import os
import torch


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds_path', type=str, required=True,
                        help='Path to predictions file')
    parser.add_argument('--targets_path', type=str, required=True,
                        help='Path to targets file')
    parser.add_argument('--preds_corrected_path', type=str, required=False,
                        help='Path to pred corrected file')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to targets file')
    return parser


def visualize(preds_path, targets_path, preds_corrected_path=None):
    preds = torch.load(preds_path, map_location='cpu')
    targets = torch.load(targets_path, map_location='cpu')
    preds_corrected = None
    if preds_corrected_path is not None:
        preds_corrected = torch.load(preds_corrected_path, map_location='cpu')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pred = preds.view(-1, 3).transpose(0, 1).detach().numpy()
    ax.plot(pred[0], pred[1], pred[2], c='r', marker='o', label='Predicted')
    target = targets.reshape(-1, 3).transpose(0, 1).detach().numpy()
    ax.plot(target[0], target[1], target[2], c='b', marker='o', label='Target')
    if preds_corrected is not None:
        pred_corr = preds_corrected.reshape(-1, 3).transpose(0, 1).detach().numpy()
        ax.plot(pred_corr[0], pred_corr[1], pred_corr[2], c='g', marker='o', label='Corrected')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_known_args()[0])
    path = args['path']
    if path is not None:
        directory = os.fsencode(path)

        # for file in os.listdir(directory):
        #     filename = os.fsdecode(file)
        #     if filename.endswith(".asm") or filename.endswith(".py"):
        #         # print(os.path.join(directory, filename))
        #         continue
        #     else:
        #         continue
        # todo
    else:
        preds_path = args['preds_path']
        targets_path = args['targets_path']
        preds_corrected_path = args['preds_corrected_path']
        # todo check paths
        visualize(preds_path, targets_path, preds_corrected_path=preds_corrected_path)

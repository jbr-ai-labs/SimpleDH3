import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
from PIL import Image as PIL_Image
import cv2
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--targets_path', type=str, required=True,
                        help='Path to targets file')
    parser.add_argument('--preds_path', type=str, required=False,
                        default=None, help='Path to predictions file')
    parser.add_argument('--preds_corrected_path1', type=str, required=False,
                        default=None, help='Path to the first pred corrected file')
    parser.add_argument('--preds_corrected_path2', type=str, required=False,
                        default=None, help='Path to the first second corrected file')
    parser.add_argument('--frames_path', type=str, required=True,
                        help='Path to save frames')
    parser.add_argument('--videos_path', type=str, required=True,
                        help='Path to save video')
    parser.add_argument('--video_name', type=str, required=True,
                        help='Name of the video without extension')
    return parser


def make_3d_frames(targets_path, name=None, preds_path=None, preds_corrected_path1=None,
                   preds_corrected_path2=None, frames_path='frames'):
    Path(frames_path).mkdir(parents=True, exist_ok=True)

    targets = torch.load(targets_path, map_location='cpu')

    preds = None
    preds_corrected1 = None
    preds_corrected2 = None
    if preds_path is not None:
        preds = torch.load(preds_path, map_location='cpu')
    if preds_corrected_path1 is not None:
        preds_corrected1 = torch.load(preds_corrected_path1, map_location='cpu')
    if preds_corrected_path2 is not None:
        preds_corrected2 = torch.load(preds_corrected_path2, map_location='cpu')

    fig = plt.figure(figsize=(9, 9))

    ax = fig.gca(projection='3d')

    target = targets.reshape(-1, 3).transpose(0, 1).detach().numpy()
    ax.plot(target[0], target[1], target[2], c='b', marker='o', label='Target')

    if preds is not None:
        pred = preds.view(-1, 3).transpose(0, 1).detach().numpy()
        ax.plot(pred[0], pred[1], pred[2], c='r', marker='o', label='Predicted')
    if preds_corrected1 is not None:
        pred_corr = preds_corrected1.reshape(-1, 3).transpose(0, 1).detach().numpy()
        ax.plot(pred_corr[0], pred_corr[1], pred_corr[2], c='g', marker='o', label='Corrected 1')
    if preds_corrected2 is not None:
        pred_corr = preds_corrected2.reshape(-1, 3).transpose(0, 1).detach().numpy()
        ax.plot(pred_corr[0], pred_corr[1], pred_corr[2], c='k', marker='o', label='Corrected 2')

    ax.legend(loc='upper right')

    ax.elev = 25.
    ax.azim = 321.
    ax.dist = 11.

    print("Rendering frames")
    for n in tqdm(range(0, 360)):
        ax.azim = ax.azim - 1
        # if n < 180:
        #    ax.dist = ax.dist-0.04
        # else:
        #    ax.dist = ax.dist+0.04

        step_str = str(n)

        if n < 10:
            step_str = '00' + step_str
        elif n < 100:
            step_str = '0' + step_str

        filename = frames_path + '/step' + step_str + '.png'
        plt.savefig(filename, bbox_inches='tight')


def render_gif(path):
    frames = []
    imgs = sorted(glob.glob('frames/*.png'))

    for i in imgs:
        new_frame = PIL_Image.open(i)
        frames.append(new_frame)

    frames[0].save(path, format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=80, loop=0)


def render_video(path, frames_dir, files_ext='png'):
    print("Making video")
    img_array = []
    for filename in tqdm(sorted(glob.glob(frames_dir + '/*.' + files_ext))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_known_args()[0])

    make_3d_frames(args['targets_path'], preds_path=args['preds_path'],
                   preds_corrected_path1=args['preds_corrected_path1'],
                   preds_corrected_path2=args['preds_corrected_path2'],
                   frames_path=args['frames_path'])

    Path(args['videos_path']).mkdir(parents=True, exist_ok=True)
    render_video(args['videos_path'] + '/' + args['video_name'] + '.avi', args['frames_path'])

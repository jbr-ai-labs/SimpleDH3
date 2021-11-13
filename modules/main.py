import logging
import argparse
import time
import random

from models.simple_architecture.simplemodel.train import simplemodel_train
from models.simple_architecture.simplemodel_coordinates.train import simplemodel_coord_train
from models.simple_architecture.data_utils import Embedding

from config_loader import load_config

MODELS_NAMES = ['simple_coordinates']


def get_logger():
    logger = logging.getLogger('Basic model (coordinates) train')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = load_config()
    file_handler = logging.FileHandler(config["PATH_TO_DEBUG_LOG"])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple_coordinates',
                        choices=MODELS_NAMES,
                        help='Which model to run')
    parser.add_argument('--use_backup', type=bool, default=False,
                        help='Use a backup or not')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model_input_size', type=int, required=True,
                        help='Model input size')
    parser.add_argument('--model_output_size', type=int, required=True,
                        help='Model output size')
    parser.add_argument('--model_hidden_dim', type=int, required=True,
                        help='Model hidden dim')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='Model hidden dim')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Number of lstm layers')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--save_prefix', type=str, help='path prefix for saving models (default: no saving)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test dataset size. Should be between 0 and 1.')
    parser.add_argument('--embedding', type=Embedding, default=Embedding.PSE,
                        help='Type of protein sequence embedding')
    parser.add_argument('--early_stopping_steps', type=int, default=None,
                        help='Number of allowed steps without improvement')
    parser.add_argument('--early_stopping_gap', type=float, default=0.01,
                        help='Minimal improvement for early stopping')
    parser.add_argument('--save_all', type=bool, default=False,
                        help='Save copy of model each epoch')
    parser.add_argument('--cross-valid', type=bool, default=False,
                        help='Run train or cross-validation data split')
    parser.add_argument('--no_test', type=bool, default=False,
                        help='Split into train/val or train/val/test')
    return parser


if __name__ == '__main__':
    n = random.randint(0, 100)
    print("Random number: ", n) 
    random.seed(n)
    main_logger = get_logger()
    main_parser = get_parser()
    args = vars(main_parser.parse_known_args()[0])
    use_backup = args['use_backup']
    choice_model = args['model']
    debug = args['debug']
    embedding = args['embedding']
    try:
        if choice_model == 'simple_coordinates':
            simplemodel_coord_train(main_logger, args, use_backup=use_backup, debug=debug, embedding=embedding)
        else:
            raise RuntimeError(f'Unknown model \'{choice_model}\'. Available models: {MODELS_NAMES}')
    except:
        if debug:
            raise
        main_logger.exception('Exception while training a model')
   

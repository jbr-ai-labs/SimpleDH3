import json
import pickle
import torch

from embedding import get_angles, get_embeddings, get_coordinates
from test_embeddings import get_embeddings_test
from esm_embeddings import get_embeddings_esm

EMBEDDINGS_NAMES = ['light', 'normal', 'esm']

if __name__ == '__main__':
    with open('config.json') as json_data_file:
        config = json.load(json_data_file)
        if config["EMBEDDINGS_NAME"] == EMBEDDINGS_NAMES[0]:
            model = torch.load(config["PATH_TO_PRETRAINED_EMBEDDING_MODEL"])
            seq = get_embeddings(config["PATH_TO_PRETRAINED_EMBEDDING_MODEL"], config["PATH_TO_SEQ_DATA"])
            with open(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_light', "wb") as seq_file:
                pickle.dump(seq, seq_file, protocol=pickle.DEFAULT_PROTOCOL)
                
            seq_test = get_embeddings(config["PATH_TO_PRETRAINED_EMBEDDING_MODEL"], config["PATH_TO_SEQ_DATA_TEST"])
            with open(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_test_light', "wb") as seq_file_test:
                pickle.dump(seq_test, seq_file_test, protocol=pickle.DEFAULT_PROTOCOL)
                
        elif config["EMBEDDINGS_NAME"] == EMBEDDINGS_NAMES[1]:
            seq = get_embeddings_test(config["PATH_TO_SEQ_DATA"])
            with open(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_normal', "wb") as seq_file:
                pickle.dump(seq, seq_file, protocol=pickle.DEFAULT_PROTOCOL)
                
            seq_test = get_embeddings_test(config["PATH_TO_SEQ_DATA_TEST"])
            with open(config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_test_normal', "wb") as seq_file_test:
                pickle.dump(seq_test, seq_file_test, protocol=pickle.DEFAULT_PROTOCOL)
                
        elif config["EMBEDDINGS_NAME"] == EMBEDDINGS_NAMES[2]:
            seq = get_embeddings_esm(config["PATH_TO_SEQ_DATA"])
            torch.save(seq, config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_esm')
            
            seq_test = get_embeddings_esm(config["PATH_TO_SEQ_DATA_TEST"])
            torch.save(seq, config["PATH_TO_SEQ_EMBEDDED"] + 'cdr_h3_seq_test_esm')
                        
        coord = get_coordinates(config["PATH_TO_COORD_DATA"])
        with open(config["PATH_TO_COORD_EMBEDDED"], "wb") as coord_file:
            pickle.dump(coord, coord_file, protocol=pickle.HIGHEST_PROTOCOL)
            
        coord_test = get_coordinates(config["PATH_TO_COORD_DATA_TEST"])
        with open(config["PATH_TO_COORD_EMBEDDED"] + 'test', "wb") as coord_file_test:
            pickle.dump(coord_test, coord_file_test, protocol=pickle.HIGHEST_PROTOCOL)


from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch

def get_embeddings_test(path_to_data):
    model_dir = Path('..')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options, weights, cuda_device=0)
    print('load model')
    raw_data = {}
    with open(path_to_data, 'r') as f:
        for line in f:
            name, seq = line.split()
            raw_data[name] = seq
    print('read')
    sequence_embeddings = {}
    for name, seq in raw_data.items():
        embedding = embedder.embed_sentence(list(seq))
        residue_embd = torch.tensor(embedding).sum(dim=0)
        sequence_embeddings[name] = residue_embd
    print('save')
    return sequence_embeddings

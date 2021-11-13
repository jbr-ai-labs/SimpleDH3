import torch
import esm
embeddings_data = list()

def get_embeddings_esm(path_to_data):
    try:
        len(get_embeddings_esm.embeddings_data)
    except AttributeError:
        get_embeddings_esm.embeddings_data = list()
    if len(get_embeddings_esm.embeddings_data) > 0:
        return get_embeddings_esm.embeddings_data
    model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    print('load model')
    raw_data = {}
    print(path_to_data)
    with open(path_to_data, 'r') as f:
        for line in f:
            name, seq = line.split()
            raw_data[name] = seq
    dataset = list(raw_data.items())
    batch_labels, batch_strs, batch_tokens = batch_converter(dataset)
    print('read')
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[34])
    token_embeddings = results["representations"][34]
    print('finish')
    sequence_embeddings = {}
    for i, (_, seq) in enumerate(dataset):
        sequence_embeddings[batch_labels[i]] = token_embeddings[i, 1:len(seq) + 1]
    print('save')
    get_embeddings_esm.embeddings_data = sequence_embeddings
    return sequence_embeddings

import urllib.request
from typing import Optional

from bs4 import BeautifulSoup
import requests
import numpy as np
import pickle
import torch
import re
import os
import random
import collections
import requests
import tqdm
from string import ascii_uppercase
from Bio.PDB import *
import numpy as np
import logging
import matplotlib.pyplot as plt
import logging
import copy
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import namedtuple

def get_id(filename: str = '080721_any_antigen__summary.tsv') -> Optional[np.ndarray]:
    with open(filename, 'r') as f:
        data = f.read().split('\n')
    return np.unique([x.split('\t')[0] for x in data[1:] if x.split('\t')[0]])


def get_cdrh3(ids: np.ndarray) -> dict:
    pattern = "</a><br>H3: <a href.*?</a><br>"
    pattern2 = '<b>Sequence</b></td><td>.*'
    cdrh3_structures = {}
    long_seqs = list()
    for n, id_ in enumerate(ids):
        cdrh3_structures[id_] = list()
        url = f"http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/cdrsearch/?pdb={id_}&CDRdef_pdb=Chothia"
        with urllib.request.FancyURLopener({}).open(url) as conn:
            content = conn.read()
        matches = re.findall(pattern, content.decode())
        for m in matches:
            m = m[:-8]
            i = -1
            while m[i - 1] != '>':
                i -= 1
            complete = m[i - 12:i - 2] != 'incomplete'
            chain_pattern = 'chain=.'
            chain = re.findall(chain_pattern, m)[0][-1]
            if complete:
                m = m[i:]
                if '...' in m:
                    break
                    long_seqs.append(id_)
                    url = f"http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/cdrviewer/?CDRdef=chothia&pdb={id_}&chain={chain}&loop=CDRH3"
                    with urllib.request.FancyURLopener({}).open(url) as f:
                        content = f.read()
                    m = re.search(pattern2, content.decode())[0]
                    m = m[:-10]
                    i = -1
                    while m[i - 1] != '>':
                        i -= 1
                m = m[i:]
                cdrh3_structures[id_].append((chain, m))
            else:
                print(n, id_, (chain, m))
        # print(n, id_, cdrh3_structures[id_])
    return cdrh3_structures


def save_sequences(structures: dict, flag: bool=False) -> None:
    if flag:
        torch.save(structures, 'sequences')

def get_rotation_matrix(b, c):
    c_norm = c/np.linalg.norm(c)
    d = np.linalg.norm([c_norm[1], c_norm[2]])
    mat1 = np.array([[1, 0, 0], [0, c_norm[2]/d, -c_norm[1]/d],
                     [0, c_norm[1]/d, c_norm[2]/d]])
    mat2 = np.array([[d, 0, -c_norm[0]], [0, 1, 0], [c_norm[0], 0, d]])
    b_norm = np.dot(mat2, np.dot(mat1, b))/np.linalg.norm(b)
    d1 = np.linalg.norm([b_norm[0], b_norm[1]])
    mat3 = np.array([[-b_norm[0]/d1, -b_norm[1]/d1, 0],
                     [b_norm[1]/d1, -b_norm[0]/d1, 0], [0, 0, 1]])
    if np.dot(mat3, b_norm)[0] <= 0:
        mat3 = np.dot(mat3, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
    return np.dot(mat3, np.dot(mat2, mat1))

def make_shift(coord, shift_vector):
    for i in range(len(coord)):
        coord[i] = coord[i] - shift_vector
    return coord, shift_vector

def change_coord(coord):
    sv = coord[-1].copy()
    coord, v = make_shift(coord, sv)
    m = get_rotation_matrix(coord[-2], coord[0])
    for i in range(len(coord)):
        coord[i] = np.dot(m, coord[i])
    return coord, v, m

abbr = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

def translate_residue(res):
    if res in abbr:
        return abbr[res]
    return '-'

def download_file(url, output_path):
    logging.basicConfig(format='%(asctime)s %(message)s', filename='pdb_log.txt', level=logging.ERROR)
    req = requests.get(url)
    if not req.ok:
        logging.error(f"Error code = {req.status_code}")
    else:
        with open(output_path, 'w') as f:
            f.write(req.content.decode('utf-8'))

def download_chothia_pdb_files(pdb_ids, antibody_database_path,
                               max_workers=4):
    """
    :param pdb_ids: A set of PDB IDs to download
    :type pdb_ids: set(str)
    :param antibody_database_path: Path to the directory to save the PDB files to.
    :type antibody_database_path: str
    :param max_workers: Max number of workers in the thread pool while downloading.
    :type max_workers: int
    """
    pdb_file_paths = [os.path.join(antibody_database_path, pdb + '.pdb') for pdb in pdb_ids]
    # Download PDBs using multiple threads
    download_url = 'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=chothia'
    urls = [download_url.format(pdb) for pdb in pdb_ids]

    for args in zip(urls, pdb_file_paths):
        download_file(args[0], args[1])

def get_rotation_matrix(b, c):
    c_norm = c/np.linalg.norm(c)
    d = np.linalg.norm([c_norm[1], c_norm[2]])
    mat1 = np.array([[1, 0, 0], [0, c_norm[2]/d, -c_norm[1]/d],
                     [0, c_norm[1]/d, c_norm[2]/d]])#x
    mat2 = np.array([[d, 0, -c_norm[0]], [0, 1, 0], [c_norm[0], 0, d]])#y
    b_norm = np.dot(mat2, np.dot(mat1, b))/np.linalg.norm(b)
    d1 = np.linalg.norm([b_norm[0], b_norm[1]])
    mat3 = np.array([[-b_norm[0]/d1, -b_norm[1]/d1, 0],
                     [b_norm[1]/d1, -b_norm[0]/d1, 0], [0, 0, 1]])#z
    if np.dot(mat3, b_norm)[0] <= 0:
        mat3 = np.dot(mat3, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))#z
    return np.dot(mat3, np.dot(mat2, mat1))

def make_shift(coord, shift_vector):
    for i in range(len(coord)):
        coord[i] = coord[i] - shift_vector
    return coord, shift_vector

def change_coord(coord):
    sv = coord[0].copy()
    coord, v = make_shift(coord, sv)
    m = get_rotation_matrix(coord[1], coord[-1])
    for i in range(len(coord)):
        coord[i] = np.dot(m, coord[i])
    return coord, v, m

def move_to_coords(coord, v, m):
    m1 = np.linalg.inv(m)
    sv = coord[0].copy()
    coord, _ = make_shift(coord, sv)
    for i in range(len(coord)):
        coord[i] = np.dot(m1, coord[i])
    coord, _ = make_shift(coord, -v)
    return coord

def make_move(coords1, coords2):
    _, v, m = change_coord(coords1)
    return move_to_coords(coords2, v, m)

def get_coords(chain, loop, ant):
    residues = list(chain.get_residues())
    sequence = ''.join([translate_residue(r.get_resname()) for r in residues])

    l_b = sequence.find(loop) - 1
    l_e = l_b + len(loop) + 2
    seq = sequence[l_b:l_e]
  
    if l_b == -1:
        print('\t', ant, chain, loop, "LOOP IS NOT FOUND")
        return
    coords = [list(map(lambda a: a.get_coord(), list(r.get_atoms())[:3])) \
              for r in residues[l_b:l_e]]
    if len(seq) != len(coords):
        print(ant)
        print(seq)
        print(residues[l_b:l_e])
    return seq, np.array(coords).reshape(-1, 3)

def add_seqs_coords(id_: str, d_ids: dict, seq_coord: dict) -> None:
    #     print('Processing', id_)
    if len(d_ids[id_]) == 0:
        print('\t Loops are incomplete')
        return

    parser = PDBParser()
    structure = parser.get_structure(id_, '../PDB/' + id_ + '.pdb')

    chains = list(structure.get_chains())
    chains_ids = [c.get_id() for c in chains]
    chains_d = {}
    for c, idx in zip(chains, chains_ids):
        chains_d[idx] = c


    index = random.randint(0, len(d_ids[id_]) - 1)
    chain_name, loop = d_ids[id_][index]

    seq, coords = get_coords(chains_d[chain_name], loop, id_)

    coords, _, _ = change_coord(coords)
    seq_coord[id_] = (seq, coords)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', filename='parser_log.txt', level=logging.DEBUG)

    ids = get_id(filename= '080721_any_antigen_summary.tsv') 
    logging.info(f'Number of structures: {len(ids)}')

    # Get CDR-H3 sequences from SAbDab
    cdrh3_structures = get_cdrh3(ids)
    save_sequences(cdrh3_structures, True)

    # Download CDR-H3 sequences from file
    cdrh3_structures = torch.load('sequences')

    # Run if there are not necessary files in ../PDB
    download_chothia_pdb_files(list(cdrh3_structures.keys()).copy(), '../PDB', max_workers=2)

    seq_coord = {}
    for id_ in cdrh3_structures.keys():
        add_seqs_coords(id_, deepcopy(cdrh3_structures), seq_coord)
    sequences_h3 = {}
    coord_h3 = {}
    for id_ in seq_coord.keys():
        sequences_h3[id_] = seq_coord[id_][0]
        coord_h3[id_] = seq_coord[id_][1]

    filename_seq = 'cdr_h3_seq_train'
    filename_coord = 'cdr_h3_coord_train'

    with open(filename_seq, 'w') as f:
        for id_ in sequences_h3.keys():
            print(id_, sequences_h3[id_], sep=' ', end='\n', file=f)
    with open(filename_coord, 'w') as f:
        for id_ in coord_h3.keys():
            print(id_, len(coord_h3[id_]), sep=' ', end='\n', file=f)
            for c in coord_h3[id_]:
                print(c[0], c[1], c[2], sep=' ', end='\n', file=f)
            print(file=f)

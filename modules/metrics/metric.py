import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time

def scalar_prod(v1, v2):
    return torch.sum(v1 * v2, dim=-1)

def distance_between_atoms(loop):
    v1 = loop[:, :-1]
    v2 = loop[:, 1:]
    return (v1 - v2).norm(dim=-1)

def angles_between_atoms(loop, lengths, on_cpu=False):
    loop = loop.reshape(loop.shape[0], -1, 3)
    a = loop[:, :-2]
    b = loop[:, 1:-1]
    c = loop[:, 2:]
    ba = a - b
    bc = c - b
    norm_ba = ba.norm(dim=-1)
    norm_bc = bc.norm(dim=-1)
    res = norm_ba.clone()
    res[res == 0] = 1
    res1 = norm_bc.clone()
    res1[res1 == 0] = 1
    res = scalar_prod(ba, bc) / res / res1
    return res

def angle(v1, v2):
    if type(v1) == torch.Tensor:
        v1 = v1.cpu()
        v1 = v1.detach().numpy()
    if type(v2) == torch.Tensor:
        v2 = v2.cpu()
        v2 = v2.detach().numpy()
    n1 = np.sqrt(sum(v1 * v1))
    n2 = np.sqrt(sum(v2 * v2))
    s = sum(v1 * v2) / (n1 * n2)
    return s #np.arccos(max(-1, min(s, 1)))
    
def cross_product(v1, v2):
    v1 = v1.tolist()
    v2 = v2.tolist()
    return np.cross(v1,v2)

def calc_dihedral(v1, v2, v3, v4):
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = cross_product(ab, cb)
    v = cross_product(db, cb)
    w = cross_product(u, v)
    a = angle(u, v)
    return a

def torsion_angles_between_atoms(loop, on_cpu=False):
    l = loop.reshape(-1, 3, 3)

    # phi  
    coords = torch.cat((l[:-1, 2, :].reshape(-1, 1, 3), l[1:]), dim=1)
    a = coords[:, :-3]
    b = coords[:, 1:-2]
    c = coords[:, 2:-1]
    d = coords[:, 3:]
    # c, n, ca, c
    phi = calc_dihedral(a, b, c, d)
    
    # psi
    coords = torch.cat((l[:-1], l[1:, 0, :].reshape(-1, 1, 3)), dim=1) 
    coords = coords[1:]
    a = coords[:, :-3]
    b = coords[:, 1:-2]
    c = coords[:, 2:-1]
    d = coords[:, 3:]
    # n, ca, ca, n
    psi = calc_dihedral(a, b, c, d)

    phi = torch.Tensor(phi)
    psi = torch.Tensor(psi)
    return torch.cat((phi, psi), dim=0).norm(dim=-1) 

def rmsd(pred, test, lengths):
    lengths = lengths*3 
    msd = torch.sum((pred - test) ** 2, dim=-1).sum(dim=-1) / lengths.float()
    return torch.sqrt(msd)

def calc_atom_dist(coord1, coord2):
    squared_dist = np.sum((np.array(coord1) - np.array(coord2)) ** 2, axis=0)
    return np.sqrt(squared_dist)

def get_dist_metrics(pred, test):
    metrics = {}
    pred = pred.reshape(pred.shape[0], -1, 3)
    test = test.reshape(pred.shape[0], -1, 3)
    sum_dist = []
    for i in range(1, len(pred)):
        dist1 = calc_atom_dist(pred[i - 1], pred[i])
        dist2 = calc_atom_dist(test[i - 1], test[i])
        sum_dist.append(abs(dist1 - dist2))
    sum_dist = np.array(sum_dist)
    metrics['dist'] = sum_dist.mean()
    metrics['dist_min'] = sum_dist.min()
    metrics['dist_max'] = sum_dist.max()
    metrics['dist_median'] = np.median(sum_dist)
    return metrics

def coordinate_metrics(pred, test, lengths, on_cpu=False):
    metrics = {}
    pred = pred.reshape(pred.shape[0], -1, 3)
    test = test.reshape(pred.shape[0], -1, 3)

    mae_batch = (pred - test).norm(dim=-1).mean(-1)
    rmsd_batch = rmsd(pred, test, lengths)
    metrics['rmsd_batch'] = rmsd_batch
    metrics['mae'] = mae_batch.mean()
    metrics['mae_min'] = mae_batch.min()
    metrics['mae_max'] = mae_batch.max()
    metrics['mae_median'] = mae_batch.median()
    metrics['rmsd'] = rmsd_batch.mean()
    metrics['rmsd_max'] = rmsd_batch.max()
    metrics['rmsd_min'] = rmsd_batch.min()
    metrics['rmsd_median'] = rmsd_batch.median()
    metrics['diff_neighbours_dist'] = torch.mean(abs(distance_between_atoms(pred) -
                                                     distance_between_atoms(test)))
    metrics['diff_angles'] = torch.mean(abs(angles_between_atoms(pred, lengths, on_cpu) -
                                            angles_between_atoms(test, lengths, on_cpu)))

    return metrics

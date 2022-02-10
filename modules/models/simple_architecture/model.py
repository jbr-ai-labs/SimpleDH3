import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils


def build_rotation_matrix(v, angle):
    v = v / v.norm()

    co = torch.cos(angle)
    si = torch.sin(angle)

    return torch.tensor([
        [co + v[0] * v[0] * (1 - co), v[0] * v[1] * (1 - co) + v[2] * si, v[0] * v[2] * (1 - co) - v[1] * si],
        [v[1] * v[0] * (1 - co) - v[2] * si, co + v[1] * v[1] * (1 - co), v[1] * v[2] * (1 - co) + v[0] * si],
        [v[2] * v[0] * (1 - co) + v[1] * si, v[2] * v[1] * (1 - co) - v[0] * si, co + v[2] * v[2] * (1 - co)]
    ])

def rotate_and_scale(x, y, z, target_angle, d, qa, qd):
    xy = y - x
    zy = y - z

    n = torch.cross(xy, zy)

    angle = torch.acos(torch.dot(xy, zy) / xy.norm() / zy.norm())
    diff = target_angle - angle

    rot_m = build_rotation_matrix(n, -diff)
    rot_m2 = build_rotation_matrix(n, 2 * angle + diff)

    yz = z - y

    yz_r = yz
    yz_r2 = yz

    if angle < qa[0] or angle > qa[1]:
        yz_r = torch.matmul(rot_m, yz)
        yz_r2 = torch.matmul(rot_m2, yz)

    if yz_r.norm() < qd[0] or yz_r.norm() > qd[1]:
        yz_r = yz_r / yz_r.norm() * d
        yz_r2 = yz_r2 / yz_r2.norm() * d

    return y + yz_r, y + yz_r2

def calc_atom_dist(coord1, coord2):
    squared_dist = np.sum((np.array(coord1) - np.array(coord2)) ** 2, axis=0)
    return np.sqrt(squared_dist)


def rescale_prediction(pred):
    d1 = 1.4555
    d2 = 1.5223
    d3 = 1.3282
    ds = [d1, d2, d3]
    d = torch.tensor(ds * len(pred)).to(pred.device)
    d = d[:len(pred) - 1]
    x1 = pred[:-1]
    x2 = pred[1:]
    alpha = d / (x1 - x2).norm(dim=-1)
    alpha = torch.unsqueeze(alpha, 1).repeat(1, 3)
    old_diff = x2 - x1
    old_diff[old_diff.norm(dim=-1) < 0.001] = 0
    diff = old_diff * alpha
    pred_new = pred[0].unsqueeze(0)
    for i in range(len(diff)):
        pred_new = torch.cat((pred_new, pred_new[-1] + diff[i].unsqueeze(0)), 0)
    return pred_new


class SimpleCharRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, bilstm=False):
        super(SimpleCharRNN, self).__init__()
        self.bilstm = bilstm
        self.device = device
        if self.bilstm:
            self.model_forward = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)
            self.model_backward = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)
        else:
            self.model = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)

    def reverse_tensor(self, x):
        x = x.permute(1, 0, 2)
        idx = [i for i in range(x.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx).to(self.device)
        inverted_tensor = x.index_select(0, idx)
        inverted_tensor = inverted_tensor.permute(1, 0, 2)
        return inverted_tensor

    def forward(self, x, lengths, hiddens=None):
        if self.bilstm:
            answer_b, lengths_b, z_b = self.model_backward(self.reverse_tensor(x), lengths, hiddens)
            answer_f, lengths_f, z_f = self.model_forward(x, lengths, hiddens, self.reverse_tensor(answer_b))
            answer, lengths, z = answer_f, lengths_f, z_f
            return answer, lengths, z
        else:
            return self.model(x, lengths, hiddens)


class SimpleCharRNNUnit(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, corrector=None):
        super(SimpleCharRNNUnit, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device

  
        self.lstmcell = nn.LSTMCell(input_size + 9, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_size)

    def init_hiddens(self, batch):
        h0 = torch.zeros(batch, self.hidden_dim).requires_grad_()
        nn.init.kaiming_uniform_(h0)

        # Initialize cell state
        c0 = torch.zeros(batch, self.hidden_dim).requires_grad_()
        nn.init.kaiming_uniform_(c0)
        return (h0, c0)

    def init_preds(self, batch):
        return torch.zeros((batch, self.output_size))

    def forward(self, x, lengths, hiddens=None, y=None):
        batch_size = x.shape[0]
        h, c = self.init_hiddens(batch_size)
        answer = torch.FloatTensor([])
        output_i = torch.zeros((x.size(0), 9))
        output_i_prev = None
        h = h.to(self.device)
        c = c.to(self.device)
        answer = answer.to(self.device)
        output_i = output_i.to(self.device)

        for i, input_i in enumerate(x.chunk(x.size(1), dim=1)):
            x_len = len(x.chunk(x.size(1), dim=1))
            if i in [0, 1, 2] or i in [x_len - 4, x_len - 3, x_len - 2, x_len - 1]:
                pass
            input_i = torch.cat((input_i, output_i.unsqueeze(1)), dim=-1)
            h, c = self.lstmcell(input_i.squeeze(1), (h, c))
            output_i = self.h2o(h)
            if y is not None:
                output_i = (output_i + y[:, i, :]) / 2
            output_i_prev = output_i.view(-1, 9).clone()
            
            if i == 0:
                for out_i in output_i:
                    out_i[0] = 0
                    out_i[1] = 0
                    out_i[2] = 0
            answer = torch.cat((answer, output_i.unsqueeze(1)), dim=1)
        return answer, lengths, None


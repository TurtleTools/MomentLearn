import numpy as np
import torch
from torch import nn
import random
import typing as ty
from MomentLearn.utils import get_score


class Net(torch.nn.Module):
    def __init__(self, number_of_bits, number_of_moments):
        super(Net, self).__init__()
        self.lin = torch.nn.Linear(number_of_moments, number_of_bits)

    def forward(self, x, y, z):
        return self.lin(x), self.lin(y), z

    def forward_single(self, x):
        return self.lin(x)


def draw_random_from_with_coords(data_moments, data_coords):
    i = random.choice(range(len(data_moments)))
    j = random.choice(range(len(data_moments[i])-data_coords[0][0].shape[0]))
    return data_moments[i][j], data_coords[i][j]


def draw_random_from_data(data: ty.List[np.ndarray]):
    return random.choice(data[random.choice(range(len(data)))])


def sample_random_moment_with_close_distant_with_coords(data_moment, data_coords, jump=5, batch=200, number_of_moments=16):
    x_all = np.zeros((batch, number_of_moments), dtype="float32")
    x_sim_dist_all = np.zeros((batch, number_of_moments), dtype="float32")
    which = np.array([random.choice([0, 1]) for _ in range(batch)]).astype("float32")
    scores = []
    for i in range(batch):
        inter_idx = random.choice(range(len(data_moment)))
        prot_len = len(data_moment[inter_idx])
        intra_idx = random.choice(range(jump, prot_len - jump - data_coords[0][0].shape[0]))
        jump_idx = random.choice(list(range(-jump, 0, 1)) + list(range(1, jump + 1, 1))) + intra_idx
        x_moment, x_coords = data_moment[inter_idx][intra_idx], data_coords[inter_idx][intra_idx]
        x_sim_moment, x_sim_coords = data_moment[inter_idx][jump_idx], data_coords[inter_idx][jump_idx]
        x_dist_moment, x_dist_coords = draw_random_from_data(data_moment, data_coords)

        if which[i] == 0:
            scores.append(get_score(x_coords, x_dist_coords))
            x_sim_dist_all[i] = x_dist_moment
        else:
            scores.append(get_score(x_coords, x_sim_coords))
            x_sim_dist_all[i] = x_sim_moment
        x_all[i] = x_moment

    scores = np.array(scores).astype("float32")
    return torch.tensor(x_all), torch.tensor(x_sim_dist_all), torch.tensor(scores)


def loss_func(out, distant, y):
    dist_sq = torch.sum(torch.pow(out - distant, 2), 1)
    dist = torch.sqrt(dist_sq + 1e-10)
    mdist = 1 - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / out.size()[0]
    return loss


def sample_random_moment_with_close_distant(data, jump=1, batch=1000, number_of_moments=16):
    x_all = np.zeros((batch, number_of_moments), dtype="float32")
    which = np.array([random.choice([0, 1]) for _ in range(batch)]).astype("float32")
    x_sim_dist_all = np.zeros((batch, number_of_moments), dtype="float32")
    for i in range(batch):
        inter_idx = random.choice(range(len(data)))
        prot_len = len(data[inter_idx])
        intra_idx = random.choice(range(jump, prot_len-jump))
        jump_idx = random.choice(list(range(-jump, 0, 1)) + list(range(1, jump+1, 1))) + intra_idx
        x, x_sim, x_dist = data[inter_idx][intra_idx], data[inter_idx][jump_idx], draw_random_from_data(data)
        x_all[i] = x
        if which[i] == 1:
            x_sim_dist_all[i] = x_sim
        else:
            x_sim_dist_all[i] = x_dist
    return torch.tensor(x_all), torch.tensor(x_sim_dist_all), torch.tensor(which)


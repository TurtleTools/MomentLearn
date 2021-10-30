import numpy as np
import torch
from torch import nn
import random
import typing as ty


class Net(nn.Module):
    def __init__(self, number_of_bits, number_of_moments):
        super(Net, self).__init__()
        self.lin = nn.Linear(number_of_moments, number_of_bits)

    def forward(self, x, y, z):
        return self.lin(x.T), self.lin(y.T), self.lin(z.T)

    def forward_single(self, x):
        return self.lin(x.T).cpu().detach().numpy()


def loss_func(out, near, distant):
    dist, near = (torch.sum(torch.abs(out - distant))), (torch.sum(torch.abs(out - near)))
    diff = near - dist
    return 1 / (torch.abs(diff) / out.shape[0])


def draw_random_from_data(data: ty.List[np.ndarray]):
    return random.choice(data[random.choice(range(len(data)))])


def sample_random_moment_with_close_distant(data: ty.List[np.ndarray], jump: int=1, batch: int=2000):
    x_all = np.zeros((batch, 16), dtype="float32")
    x_sim_all = np.zeros((batch, 16), dtype="float32")
    x_dist_all = np.zeros((batch, 16), dtype="float32")
    for i in range(batch):
        inter_idx = random.choice(range(len(data)))
        prot_len = len(data[inter_idx])
        intra_idx = random.choice(range(jump, prot_len - jump))
        jump_idx = random.choice(range(jump, jump + 1, 1)) + intra_idx
        x, x_sim, x_dist = data[inter_idx][intra_idx], data[inter_idx][jump_idx], draw_random_from_data(data)
        x_all[i] = x
        x_sim_all[i] = x_sim
        x_dist_all[i] = x_dist
    return torch.tensor(x_all.T), torch.tensor(x_sim_all.T), torch.tensor(x_dist_all.T)

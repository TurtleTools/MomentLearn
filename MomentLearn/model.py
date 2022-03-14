import numpy as np
import torch
import random
import typing as ty
from MomentLearn.utils import get_score


class LinearMoment(torch.nn.Module):
    def __init__(self, number_of_moments, output_size):
        super(LinearMoment, self).__init__()
        self.lin = torch.nn.Linear(number_of_moments, output_size)

    def forward(self, x):
        return self.lin(x)


class ContrastiveLearn(torch.nn.Module):
    def __init__(self, number_of_moments, output_size):
        super(ContrastiveLearn, self).__init__()
        self.linear_moment = LinearMoment(number_of_moments, output_size)

    def forward(self, x, y, z):
        return self.linear_moment(x), self.linear_moment(y), z

    def forward_single(self, x):
        return self.linear_moment(x)


class MomentLearn(torch.nn.Module):
    def __init__(self, number_of_moments, cont_dim):
        super(MomentLearn, self).__init__()
        self.linear_segment = torch.nn.Sequential(
            torch.nn.Linear(number_of_moments, number_of_moments),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(number_of_moments),
            torch.nn.Linear(number_of_moments, cont_dim),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(cont_dim)
        )


    def forward(self, x, y, z):
        return self.linear_segment(x), self.linear_segment(y), z

    def forward_single_lab(self, x):
        return self.linear_segment(x)

    def forward_single_segment(self, x):
        return self.linear_segment(x)



class GRUMoment(torch.nn.Module):
    def __init__(self, input_size, number_of_moments, hidden_size, layer, batch_size):
        super(GRUMoment, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_of_moments = number_of_moments
        self.input_size = input_size
        self.layer = layer
        self.linear_moment = LinearMoment(number_of_moments, input_size)
        self.rnn = torch.nn.GRU(input_size, hidden_size, layer, batch_first=True)

    def forward(self, x, y, h_01, h_02, sizesx, sizesy):
        x, hx = self.forward_single(x, h_01, sizesx)
        y, hy = self.forward_single(y, h_02, sizesy)
        return (x, hx), (y, hy)

    def forward_single(self, x, h_0, sizes):
        x = self.linear_moment(x)
        x = torch.nn.utils.rnn.pad_sequence(
            [x[sizes[:i-1].sum(): sizes[:i].sum()] for i in range(1, sizes.shape[0] + 1)], batch_first=True)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sizes, batch_first=True)
        x, h_n = self.rnn(x, h_0)
        return x, h_n

    def init_hidden(self):
        return torch.zeros(self.layer, self.batch_size, self.hidden_size)


class LSTMMoment(torch.nn.Module):
    def __init__(self, input_size, number_of_moments, hidden_size, layer, batch_size):
        super(LSTMMoment, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_of_moments = number_of_moments
        self.input_size = input_size
        self.layer = layer
        self.linear_moment = LinearMoment(number_of_moments, input_size)
        self.rnn = torch.nn.LSTM(input_size, hidden_size, layer, batch_first=True)

    def forward(self, x, y, h_01, h_02, c_01, c_02, sizesx, sizesy):
        x, (hx, cx) = self.forward_single(x, h_01, c_01, sizesx)
        y, (hy, cy) = self.forward_single(y, h_02, c_02, sizesy)
        return (x, (hx, cx)), (y, (hy, cy))

    def forward_single(self, x, h_0, c_0, sizes):
        x = self.linear_moment(x)
        x = torch.nn.utils.rnn.pad_sequence(
            [x[sizes[:i-1].sum(): sizes[:i].sum()] for i in range(1, sizes.shape[0] + 1)], batch_first=True)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sizes, batch_first=True)
        x, (h_n, c_n) = self.rnn(x, (h_0, c_0))
        return x, (h_n, c_n)

    def init_hidden(self):
        return (torch.zeros(self.layer, self.batch_size, self.hidden_size),
                torch.zeros(self.layer, self.batch_size, self.hidden_size))


def draw_random_from_with_coords(data_moments, data_coords):
    i = random.choice(range(len(data_moments)))
    j = random.choice(range(len(data_moments[i])-data_coords[0][0].shape[0]))
    return data_moments[i][j], data_coords[i][j]


def draw_random_from_data(data: ty.List[np.ndarray]):
    return random.choice(data[random.choice(range(len(data)))])


def sample_double_proteins_with_sim_dist(data_moment, classes, batch=200):
    selected_proteins1, sizes1 = [], []
    selected_proteins2, sizes2 = [], []
    which = np.array([random.choice([0, 1]) for _ in range(batch)]).astype("float32")
    for i in range(batch):
        inter_idx = random.choice(range(len(data_moment)))
        protein_class = classes[inter_idx]
        selected_proteins1.append(data_moment[inter_idx])
        sizes1.append(len(selected_proteins1[-1]))
        if which[i] == 1:
            other_idx = random.choice(list(np.where(classes == protein_class)[0]))
            # other_idx = inter_idx
        else:
            other_idx = random.choice(list(np.where(classes != protein_class)[0]))
        selected_proteins2.append(data_moment[other_idx])
        sizes2.append(len(selected_proteins2[-1]))
    sort_idx1 = np.argsort(sizes1)[::-1]
    sort_idx2 = np.argsort(sizes2)[::-1]
    selected_proteins1 = torch.tensor(np.concatenate(np.array(selected_proteins1)[sort_idx1]).astype("float32"))
    selected_proteins2 = torch.tensor(np.concatenate(np.array(selected_proteins2)[sort_idx2]).astype("float32"))

    return ((selected_proteins1, np.array(sizes1)[sort_idx1], sort_idx1),
            (selected_proteins2, np.array(sizes2)[sort_idx2], sort_idx2),
            torch.tensor(which))


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


def sample_random_moment_with_same_distant(data, jump=1, batch=1000, number_of_moments=16):
    x_all = np.zeros((batch, number_of_moments), dtype="float32")
    which = np.array([random.choice([0, 1]) for _ in range(batch)]).astype("float32")
    x_sim_dist_all = np.zeros((batch, number_of_moments), dtype="float32")
    for i in range(batch):
        inter_idx = random.choice(range(len(data)))
        prot_len = len(data[inter_idx])
        intra_idx = random.choice(range(jump, prot_len-jump))
        x, x_sim, x_dist = data[inter_idx][intra_idx], data[inter_idx][intra_idx], draw_random_from_data(data)
        x_all[i] = x
        if which[i] == 1:
            x_sim_dist_all[i] = x_sim
        else:
            x_sim_dist_all[i] = x_dist
    return torch.tensor(x_all), torch.tensor(x_sim_dist_all), torch.tensor(which)


def sample_random_moment_with_close_further(data, jump=1, batch=1000, number_of_moments=16, further_min=5, further_max=20):
    x_all = np.zeros((batch, number_of_moments), dtype="float32")
    which = np.array([random.choice([0, 1]) for _ in range(batch)]).astype("float32")
    x_sim_dist_all = np.zeros((batch, number_of_moments), dtype="float32")
    for i in range(batch):
        inter_idx = random.choice(range(len(data)))
        prot_len = len(data[inter_idx])
        intra_idx = random.choice(range(jump, prot_len-jump))
        jump_idx = random.choice(list(range(-jump, 0, 1)) + list(range(1, jump+1, 1))) + intra_idx
        jump_idx2 = random.choice(list(range(max(0, jump_idx - further_max), max(0, jump_idx - further_min))) + list(range(min(jump + further_min, prot_len), min(prot_len, jump + further_max))))
        x, x_sim, x_dist = data[inter_idx][intra_idx], data[inter_idx][jump_idx], data[inter_idx][jump_idx2]
        x_all[i] = x
        if which[i] == 1:
            x_sim_dist_all[i] = x_sim
        else:
            x_sim_dist_all[i] = x_dist
    return torch.tensor(x_all), torch.tensor(x_sim_dist_all), torch.tensor(which)
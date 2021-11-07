import pandas as pnd
from MomentLearn import utils
from MomentLearn import model as model_utils
from MomentLearn.model import ContrastiveLearn
from geometricus import MomentType
import torch
import numpy as np


def main():
    X_names, _ = utils.get_example_metadata()
    kmer_size = 25
    moment_types = list(MomentType)

    data = list(utils.get_all_kmer_moments_for_pdbs(X_names, kmer_size=kmer_size,
                                                    moment_types=moment_types))
    output_dim = 1
    model = ContrastiveLearn(output_dim, len(moment_types))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    epoch = 100_000

    current_losses = []
    for e in range(epoch):
        x, dist, y = model_utils.sample_random_moment_with_close_distant(data,
                                                                         batch=300,
                                                                         number_of_moments=len(moment_types))
        x, dist, y = model(x, dist, y)

        loss = model_utils.loss_func(x, dist, y)
        optimizer.zero_grad()
        loss.backward()
        current_losses.append(loss.item())
        optimizer.step()

        if e % 1000 == 0:
            print(np.mean(current_losses))
            current_losses = []

    torch.save(model, "model.pth")


if __name__ == "__main__":
    main()
import pandas as pnd
from MomentLearn import utils
from MomentLearn import model as model_utils
from MomentLearn.model import Net
from geometricus import MomentType
import torch
import numpy as np


def main():

    url = "https://raw.githubusercontent.com/TurtleTools/geometricus/master/example_data/MAPK_KLIFS.tsv"
    mapk_df = pnd.read_csv(url, sep="\t")
    mapk_pdb_id_to_class = {}
    for pdb_id, chain, class_name in list(zip(mapk_df["PDB"], mapk_df["CHAIN"], mapk_df["CLASS"])):
        mapk_pdb_id_to_class[(pdb_id, chain)] = class_name
    len(mapk_pdb_id_to_class)
    X_names = list(mapk_pdb_id_to_class.keys())

    kmer_size = 16
    moment_types = list(MomentType)

    data = list(utils.get_all_kmer_moments_for_pdbs(X_names, kmer_size=kmer_size,
                                               moment_types=moment_types))

    output_dim = 1
    model = Net(output_dim, len(moment_types))
    optimizer = torch.optim.Adam(model.parameters(), lr=2)
    epoch = 100000

    current_losses = []
    for e in range(epoch):
        x, x_sim, x_dist = model_utils.sample_random_moment_with_close_distant(data, batch=1000)
        x, y, z = model(x, x_sim, x_dist)
        loss = model_utils.loss_func(x, y, z)
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
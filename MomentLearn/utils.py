import typing as ty
from geometricus import MomentInvariants, SplitType, MomentType
import prody as pd
import numpy as np
from caretta import superposition_functions
from caretta import score_functions
import umap
import torch


def get_all_kmer_moments_for_pdbs(pdbs: ty.List[ty.Tuple[str, str]], kmer_size: int=16,
                                  moment_types: ty.Union[None, ty.List[MomentType]]=None):
    if moment_types is None:
        moment_types = list(MomentType)
    for pdb, chain in pdbs:
        atom_group: pd.AtomGroup = pd.parsePDB(pdb, chain=chain)
        invariant = MomentInvariants.from_prody_atomgroup(pdb,
                                                    atom_group,
                                                    split_type=SplitType.KMER,
                                                    split_size=16, moment_types=moment_types)
        yield ((np.sign(invariant.moments) * np.log1p(np.abs(invariant.moments))) / kmer_size).astype("float32")


def get_score(coords1, coords2):
    rot2, trans2 = superposition_functions.svd_superimpose(coords1, coords2)
    rot1, trans1 = superposition_functions.svd_superimpose(coords2, coords1)
    coords1, coords2 = superposition_functions.apply_rotran(coords1, rot1, trans1), superposition_functions.apply_rotran(coords2, rot2, trans2)
    return score_functions.get_caretta_score(coords1, coords2, gamma=.01).mean().round(4)

def get_example_metadata():
    import pandas as pnd
    url = "https://raw.githubusercontent.com/TurtleTools/geometricus/master/example_data/MAPK_KLIFS.tsv"
    mapk_df = pnd.read_csv(url, sep="\t")
    mapk_pdb_id_to_class = {}
    for pdb_id, chain, class_name in list(zip(mapk_df["PDB"], mapk_df["CHAIN"], mapk_df["CLASS"])):
        mapk_pdb_id_to_class[(pdb_id, chain)] = class_name
    X_names = list(mapk_pdb_id_to_class.keys())
    return X_names, mapk_pdb_id_to_class

def get_embedding(prot_rep, bins):
    return np.histogram(prot_rep, bins=bins)[0]

def plot_umap(protein_moments,
              model,
              protein_labels):
    all_moments = np.concatenate(protein_moments)
    import matplotlib.pyplot as plt
    ms = torch.tensor(all_moments.astype("float32"))
    r, _, _ = model.forward(ms,ms,ms)
    r = r.cpu().detach().numpy().T.flatten()
    h = plt.hist(r)
    start, end = h[1][0] , h[1][-1]
    split = (h[1][-1] -  h[1][0])/100

    ind_moments_compressed = [model.forward(torch.tensor(x.astype("float32")),
                                            torch.tensor(x.astype("float32")),
                                            torch.tensor(x.astype("float32")))[0].cpu().detach().numpy().T.flatten() for x in protein_moments]

    protein_embeddings = [get_embedding(x, bins=np.arange(start, end, split)) for x in ind_moments_compressed]
    protein_embeddings = [x/x.sum() for x in protein_embeddings]

    reducer = umap.UMAP(metric="euclidean", n_components=2, n_neighbors=20)
    reduced = reducer.fit_transform(protein_embeddings)

    plt.figure(figsize=(10,10))
    for i in range(3):
        indices = np.where(np.array(protein_labels) == i)[0]
        plt.scatter(reduced[indices, 0],
                    reduced[indices, 1],
                    edgecolor="black",
                    linewidth=0.1,
                    alpha=0.8)
    plt.axis("off")
    plt.legend()
    plt.show()
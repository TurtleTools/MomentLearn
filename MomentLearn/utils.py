import imp
import typing as ty
from geometricus import MomentInvariants, SplitType, MomentType
import prody as pd
import numpy as np
from caretta import superposition_functions
from caretta import score_functions
import umap
import torch
from collections import OrderedDict
from sklearn.metrics import pairwise_distances


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


def moment_tensors_to_bits(list_of_moment_tensors, nbits=16):
    bits = []
    for segment in list_of_moment_tensors:
        x = (segment < 0).astype("uint8")
        x = np.packbits(x)
        bits.append(x)
    return np.array(bits, dtype="uint8").view(f"|S{nbits//8}").flatten()


def mol_to_segs(mol, steps=1, k=175):
    segs = []
    for i in range(0, len(mol) - k + 1, steps):
        segs.append(mol[i: i + k])
    return np.array(segs).astype(np.float32)


def moments_to_tensors(segments, model):
    return model.forward_single_segment(torch.tensor(segments)).cpu().detach().numpy()


def moments_to_bit_list(list_of_moments, model, nbits=16):
    moment_tensors = model.forward_single_segment(torch.tensor(list_of_moments)).cpu().detach().numpy()
    return list(moment_tensors_to_bits(moment_tensors, nbits=nbits))


def get_all_keys(list_of_moment_hashes, model, nbits=16):
    all_keys = set()
    for prot in list_of_moment_hashes:
        all_keys |= set(moments_to_bit_list(prot, model, nbits=nbits))
    return list(all_keys)


def count_with_keys(prot_hashes, keys):
    d = OrderedDict.fromkeys(keys, 0)
    for hash in prot_hashes:
        d[hash] += 1
    return np.array([d[x] for x in keys])


def get_hash_embeddings(protein_moments, model, nbits=16):
    ind_moments_compressed = [moments_to_bit_list(x, model, nbits=nbits) for x in protein_moments]
    all_keys = get_all_keys(protein_moments, model, nbits=nbits)
    print(len(all_keys))
    protein_embeddings = [count_with_keys(x, all_keys) for x in ind_moments_compressed]
    return [x/x.sum() for x in protein_embeddings]


def get_distmat(protein_moments, model, nbits=16):
    embeddings = np.array(get_hash_embeddings(protein_moments, model, nbits=nbits))
    return pairwise_distances(embeddings, metric="l1")

def get_dist_mat_raw(protein_moments, model):
    return [moments_to_tensors(x, model) for x in protein_moments]

def plot_umap(protein_moments,
              model,
              protein_labels,
              nbits=8):
    import matplotlib.pyplot as plt

    protein_embeddings = get_hash_embeddings(protein_moments, model, nbits=nbits)

    reducer = umap.UMAP(metric="euclidean", n_components=2, n_neighbors=20)
    reduced = reducer.fit_transform(protein_embeddings)

    plt.figure(figsize=(10,10))
    for i in range(len(set(protein_labels))):
        indices = np.where(np.array(protein_labels) == i)[0]
        plt.scatter(reduced[indices, 0],
                    reduced[indices, 1],
                    edgecolor="black",
                    linewidth=0.1,
                    alpha=0.8)
    plt.axis("off")
    plt.legend()
    plt.show()
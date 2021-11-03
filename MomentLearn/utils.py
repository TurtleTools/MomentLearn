import typing as ty
from geometricus import MomentInvariants, SplitType, MomentType
import prody as pd
import numpy as np
from caretta import superposition_functions
from caretta import score_functions


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
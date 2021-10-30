import typing as ty
from geometricus import MomentInvariants, SplitType, MomentType
import prody as pd
import numpy as np


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

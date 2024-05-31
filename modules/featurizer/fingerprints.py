import numpy as np
from mordred import Calculator, descriptors

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

__all__ = ["morganfp", "RDKfp", "stacked_RDKfp", "mordred"]


def morganfp(x, n=2048, radii=2):
    fp3 = AllChem.GetMorganFingerprintAsBitVect(x, radii, nBits=n)
    res3 = np.zeros(len(fp3))
    DataStructs.ConvertToNumpyArray(fp3, res3)
    return res3


def RDKfp(x, n=2048, length=5):
    fp3 = Chem.RDKFingerprint(x, maxPath=length, fpSize=n)
    res3 = np.zeros(len(fp3))
    DataStructs.ConvertToNumpyArray(fp3, res3)
    return res3


def stacked_RDKfp(x, n=2048, lengths=(5, 7, 9, 11)):
    res = []
    for l in lengths:
        res.append(RDKfp(x, n=n, length=l))
    return np.concatenate(res)


def mordred(mol, ignore_3d=True):
    calc = Calculator(descriptors, ignore_3D=ignore_3d)
    desc = calc(mol)
    return np.array(desc)

from collections import OrderedDict

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from rdkit import Chem

from modules.featurizer import fingerprints
from .featurizer import Featurizer


def matrices2mol(node_labels, edge_labels):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom.SetAtomicNum(node_label))

    for start, end in zip(*np.nonzero(edge_labels)):
        if start > end:
            mol.AddBond(int(start), int(end), edge_labels[start, end])
    return mol


def stack_dict(features, keys, get_column_names=False):
    names = []
    stacked_features = []
    for name in keys:
        feature = np.array(features[name]).reshape(-1)
        stacked_features.append(feature)
        if get_column_names:
            if len(feature) > 1:
                col_names = [f"{name}_{i}" for i in range(len(feature))]
                names += col_names
            else:
                names.append(name)
    return np.concatenate(stacked_features), names


class Molecule(Featurizer):
    def __init__(self, smiles, external_features=None, precomputed_features=None, drop_hs=True,
            graph_features=True, linear_features=True,
            fingerprint_names=("morganfp", "RDKfp", "stacked_RDKfp", "mordred")):
        """
        This is a class of small molecule description.
        Args:
            smiles: str: Input smiles string.
            external_features: dict, optional (default=None): Dictionary of external features,
            properties, etc.
            precomputed_features: dict, optional (default=None): Dictionary of precomputed features.
            drop_hs: bool: Drop hydrogens.
        """
        mol = Chem.MolFromSmiles(smiles)
        self.graph_features = graph_features
        self.linear_features = linear_features
        self.fingerprint_names = fingerprint_names

        self.arguments = {"drop_hs": drop_hs, "graph_features": graph_features,
                          "linear_features": linear_features}
        if drop_hs:
            self.rdkit_molecule = Chem.RemoveHs(mol)
        else:
            self.rdkit_molecule = Chem.AddHs(mol, explicitOnly=True)

        self.rdkit_molecule.UpdatePropertyCache(strict=False)
        self.smiles = smiles

        if external_features is None:
            external_features = {}

        if precomputed_features is None:
            features = self.calculate_features()
            self.features = {**features, **external_features}
        else:
            self.features = precomputed_features

    def calculate_features(self):
        features = {}
        if self.graph_features:
            features = {**features, **self.calculate_graph_features()}
        if self.linear_features:
            features = {**features, **self.calculate_linear_features()}
        return features

    def calculate_linear_features(self):
        fingerprint_dict = {}
        for fingerprint in self.fingerprint_names:
            fingerprint_dict[fingerprint] = getattr(fingerprints, fingerprint)(self.rdkit_molecule)
        return fingerprint_dict

    def calculate_graph_features(self):
        """

        Returns: dict: Calculated features.

        """
        senders, recievers, types = self.get_senders_recievers_types()
        edge_features = {"sender": senders, "reciever": recievers, "edge_type": types}
        atom_types = self.get_node_types()
        atom_features = self.get_atomic_features()
        atom_features["atomic_numbers"] = atom_types
        features = {"atomic_features": atom_features,
                    "edge_features": edge_features,
                    "smiles": self.smiles,
                    "rdmol": self.rdkit_molecule}
        return features

    def get_senders_recievers_types(self):
        senders = [b.GetBeginAtomIdx() for b in self.rdkit_molecule.GetBonds()]
        receivers = [b.GetEndAtomIdx() for b in self.rdkit_molecule.GetBonds()]
        b_types = ['AROMATIC' if b.GetIsAromatic() else str(b.GetBondType()) for b in
                   self.rdkit_molecule.GetBonds()]
        return (np.array(senders, dtype=np.int32),
                np.array(receivers, dtype=np.int32),
                np.array(b_types, dtype='<U10'))

    def get_node_types(self):
        return np.array([atom.GetAtomicNum() for atom in self.rdkit_molecule.GetAtoms()],
                        dtype=np.int32)

    def get_num_atoms(self):
        return self.rdkit_molecule.GetNumAtoms()

    def get_adjacency_matrix(self):
        senders, receivers, b_types = self.get_senders_recievers_types()
        num_atoms = self.get_num_atoms()
        adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype='<U10')
        adjacency_matrix[senders, receivers] = b_types
        adjacency_matrix[receivers, senders] = b_types
        return adjacency_matrix

    def get_smiles(self):
        return Chem.MolToSmiles(self.rdkit_molecule)

    def get_atomic_features(self):

        degrees = [a.GetDegree() for a in self.rdkit_molecule.GetAtoms()]
        exp_valences = [a.GetExplicitValence() for a in self.rdkit_molecule.GetAtoms()]
        hybridization = [int(a.GetHybridization()) for a in self.rdkit_molecule.GetAtoms()]
        imp_valences = [a.GetImplicitValence() for a in self.rdkit_molecule.GetAtoms()]
        is_aromatic = [a.GetIsAromatic() for a in self.rdkit_molecule.GetAtoms()]
        num_explicit_hs = [a.GetNumExplicitHs() for a in self.rdkit_molecule.GetAtoms()]
        num_implicit_hs = [a.GetNumImplicitHs() for a in self.rdkit_molecule.GetAtoms()]
        is_ring = [a.IsInRing() for a in self.rdkit_molecule.GetAtoms()]
        num_radical_electrons = [a.GetNumRadicalElectrons() for a in self.rdkit_molecule.GetAtoms()]
        formal_charge = [a.GetFormalCharge() for a in self.rdkit_molecule.GetAtoms()]

        features = [degrees, exp_valences, hybridization, imp_valences, is_aromatic,
                    num_explicit_hs, num_implicit_hs, is_ring, num_radical_electrons, formal_charge]
        features = [np.array(i, np.int32) for i in features]

        feature_names = ['degree', 'explicit_valence', 'hybridization', 'implicit_valence',
                         'is_aromatic', 'num_explicit_hs', 'num_implicit_hs', 'in_ring',
                         'num_radical_electrons', 'formal_charge']
        features = dict(zip(feature_names, features))
        features["num_hs"] = features["num_explicit_hs"] + features["num_implicit_hs"]
        features["valence"] = features["implicit_valence"] + features["explicit_valence"]
        return features

    def get_atoms_mapping(self):
        map_num = [atom.GetAtomMapNum() for atom in self.rdkit_molecule.GetAtoms()]
        return np.array(map_num, dtype=np.int32)

    def get_degrees(self):
        return np.count_nonzero(self.get_adjacency_matrix(), -1)


class MoleculeList(Featurizer):
    def __init__(self, smiles_list, external_features_list=None,
            precomputed_features=None,
            **molecule_args):
        """

        Args:
            smiles_list: List of st: List of smiles strings.
            external_features_list:
            precomputed_features:
            **molecule_args:
        """
        self.arguments = molecule_args
        if precomputed_features is None:
            # todo: progress bar
            def calc_molecule(arg):
                mol = Molecule(*arg, **molecule_args)
                return mol

            if external_features_list is not None:
                args = list(zip(smiles_list, external_features_list))
            else:
                args = list(zip(smiles_list))
            pool = Pool()
            self.mol_list = list(pool.imap(calc_molecule, args))
            pool.close()
            pool.join()

            self.features = self.calculate_features()

        else:
            self.features = precomputed_features
        self.meta = self.compute_meta()

    def calculate_features(self):
        features = [i.features for i in self.mol_list]
        return features

    def create_feature_table(self, features_names=None, stack_features=False):
        all_features = set(self.features[0].keys())
        if features_names is None:
            features_names = all_features
        if stack_features:
            _, col_names = stack_dict(self.features[0], features_names, get_column_names=True)
            data = np.stack([stack_dict(i, features_names)[0] for i in self.features])
            return pd.DataFrame(data=data, columns=col_names)
        else:
            return pd.DataFrame.from_records(self.features,
                                             exclude=all_features - set(features_names))

    def compute_meta(self, ):
        meta = {'atomic_features': OrderedDict(),
                'edge_features': OrderedDict()}

        for val in self.features:
            for field in ["atomic_features", "edge_features"]:
                for k, v in val[field].items():
                    if k in meta[field]:
                        meta[field][k] |= set(v)
                    else:
                        meta[field][k] = {*v}
        for field in ["atomic_features", "edge_features"]:
            for k, v in meta[field].items():
                meta[field][k] = list(v)
        return meta

    def get_meta(self):
        return self.meta

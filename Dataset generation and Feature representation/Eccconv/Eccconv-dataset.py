# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:17:54 2023

@author: longzheng
"""


import os
import os.path as osp

import numpy as np
from joblib import Parallel, delayed
from tensorflow.keras.utils import get_file
from tqdm import tqdm

from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot, sparse
from spektral.utils.io import load_csv, load_sdf

ATOM_TYPES = [1, 6, 7, 8, 9, 10, 14, 15, 16, 17, 35] 
BOND_TYPES = [1, 2, 3, 4]


class Ecc_DataSet(Dataset):
    """
    In this dataset, nodes represent atoms and edges represent chemical bonds.
    There are 11 possible atom types (H, C, N, O, F, P, S, Si, Cl, Br, *) and 
    4 bond types (single,double, triple, aromatic).
    Node features represent the chemical properties of each atom and include:
    - The atomic number, one-hot encoded;
    - The atom's position in the X, Y, and Z dimensions;
    - The atomic charge;
    - The mass difference from the monoisotope;
    The edge features represent the type of chemical bond between two atoms,
    one-hot encoded.
    """

    def __init__(self, amount=None, n_jobs=1, **kwargs):
        self.amount = amount
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    def read(self):
        print("Loading Ecc_DataSet .")
        data = load_sdf('data.sdf', amount=None)  # Internal SDF format
        
        def read_mol(mol):
            x = np.array([atom_to_feature(atom) for atom in mol["atoms"]])
            a, e = mol_to_adj(mol)
            return x, a, e

        data = Parallel(n_jobs=self.n_jobs)(
            delayed(read_mol)(mol) for mol in tqdm(data, ncols=80)
        )
        x_list, a_list, e_list = list(zip(*data))

        # Load labels
        labels = load_csv('data.csv',sep=',',encoding='gbk')
        labels = labels.set_index("PID").values
        if self.amount is not None:
            labels = labels[: self.amount]

        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]
    
def atom_to_feature(atom):
   
    atomic_num = label_to_one_hot(atom["atomic_num"], ATOM_TYPES)
    coords = atom["coords"]
    charge = atom["charge"]
    iso = atom["iso"]

    return np.concatenate((atomic_num, coords, [charge, iso]), -1)


def mol_to_adj(mol):
    row, col, edge_features = [], [], []
    for bond in mol["bonds"]:
        start, end = bond["start_atom"], bond["end_atom"]
        row += [start, end]
        col += [end, start]
        edge_features += [bond["type"]] * 2

    a, e = sparse.edge_index_to_matrix(
        edge_index=np.array((row, col)).T,
        edge_weight=np.ones_like(row),
        edge_features=label_to_one_hot(edge_features, BOND_TYPES),
    )

    return a, e

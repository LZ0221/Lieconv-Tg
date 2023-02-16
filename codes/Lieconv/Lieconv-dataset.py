# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:44:16 2023

@author: longzheng
"""


import logging
import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from oil.datasetup.datasets import EasyIMGDataset
from lie_conv.hamiltonian import HamiltonianDynamics, KeplerH, SpringH
from lie_conv.lieGroups import SO3
from torchdiffeq import odeint_adjoint as odeint
from corm_data.utils import initialize_datasets
import torchvision
from itertools import islice
from math import inf
import tarfile
from torch.nn.utils.rnn import pad_sequence

charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9,'*': 10,'Si': 14,'P': 15,'S': 16,'Cl': 17,
               'Br': 35}

def process_xyz_files(data, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.
    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    logging.info('Processing data file: {}'.format(data))
    if tarfile.is_tarfile(data):
        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)
  
    else:
        raise ValueError('Can only read from directory or tarball archive!')

    # Use only files that end with specified extension.
    if file_ext is not None:
        files = [file for file in files if file.endswith(file_ext)]

    # Use only files that match desired filter.
    if file_idx_list is not None:
        files = [file for idx, file in enumerate(files) if idx in file_idx_list]

    molecules = []

    for file in files:
        with readfile(file) as openfile:
            xyz_lines = [line.decode('UTF-8') for line in openfile.readlines()]
            num_atoms = int(xyz_lines[0])
            mol_props = xyz_lines[1].split()
            mol_xyz = xyz_lines[2:num_atoms+2]
            atom_charges, atom_positions = [], []
            for line in mol_xyz:
                atom, posx, posy, posz = line.replace('*^', 'e').split()
                atom_charges.append(charge_dict[atom])
                atom_positions.append([float(posx), float(posy), float(posz)])

            prop_strings = ['index','tg', 'mw', 'tag']
            prop_strings = prop_strings[:3]
            mol_props = [int(mol_props[0])] + [float(x) for x in mol_props[1:3]]
            mol_props = dict(zip(prop_strings, mol_props))

            molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
            molecule.update(mol_props)
            molecule = {key: torch.tensor(val) for key, val in molecule.items()}
            
            molecules.append(molecule)
            

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props}

    # If stacking is desireable, pad and then stack.
    if stack:
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}

    return molecules

def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.
    Includes a check that each split contains examples of every species in the
    entire dataset.
    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset
    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.
    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}
    return all_species

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.
    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() 
                      if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}
        self.median_stats = {key: (val.median(), torch.median(torch.abs(val - val.median()))) for key, val in self.data.items() 
                             if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}



def initialize_datasets(splits):
    """
    Initialize datasets.
    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.
    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': splits[0],
                'test': splits[1], 'valid': splits[2]}

    datafiles = {'train': 'train.npz' , 'valid' : 'valid.npz' , 'test' : 'test.npz'}

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species) for split, data in datasets.items()}


    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    return datasets, num_species, max_charge
def Tgdatasets():
    datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1),)
    filename = 'data.pz'
    torch.save((datasets, num_species, charge_scale),filename)
    return (datasets, num_species, charge_scale)
    
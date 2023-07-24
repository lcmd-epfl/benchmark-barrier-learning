#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from smilify import smilify


def clean_mapping(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def get_atoms_in_order(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(mol)
    atoms   = np.array([at.GetSymbol() for at in mol.GetAtoms()])
    mapping = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
    assert np.all(mapping > 0)
    atoms, mapping = map(np.array, zip(*[(at.GetSymbol(), at.GetAtomMapNum()) for at in mol.GetAtoms()]))
    return atoms[np.argsort(mapping)]


def main(data_dir='data/proparg'):
    df = pd.read_csv(f'{data_dir}/data.csv', index_col=0)
    labels = df['mol'].values
    enans  = df['enan'].values

    for label, enan in tqdm(zip(labels, enans), total=len(df)):
        xyzs = [f'{data_dir}/data_{comp}_xyz/{label}{enan}.xyz' for comp in ('react', 'prod')]
        smis_mapped = [smilify(xyz, mapping=True) for xyz in xyzs]
        try:
            arm, apm = map(get_atoms_in_order, smis_mapped)
            assert np.all(arm==apm)
            rsmi_mapped = smis_mapped[0]+'>>'+smis_mapped[1]
            smis = [clean_mapping(smi) for smi in smis_mapped]
            rsmi = smis[0]+'>>'+smis[1]
        except:
            print('\n' + '\033[1;91m' + f'{label}{enan} is bad' + '\033[0m')
            rsmi = ''
            rsmi_mapped = ''
        df.loc[(df['mol']==label) & (df['enan']==enan), 'rxn_smiles_mapped'] = rsmi_mapped
        df.loc[(df['mol']==label) & (df['enan']==enan), 'rxn_smiles'] = rsmi
    df.to_csv('proparg-mapped.csv')


if __name__=='__main__':
    main()

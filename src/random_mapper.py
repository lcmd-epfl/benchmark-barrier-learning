import argparse as ap
from rdkit import Chem
import pandas as pd
import numpy as np

def argparse():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    parser.add_argument('-p', '--proparg', action='store_true')
    parser.add_argument('--proparg_good', action='store_true')
    args = parser.parse_args()
    return args

def set_random_atom_map(mol, seed=2):
    init_maps = get_init_maps(mol)
    print('initial maps', init_maps)
    np.random.seed(seed)
    np.random.shuffle(init_maps)
    print('random maps', init_maps)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(init_maps[i])

    return mol

def clear_atom_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol

def set_atom_map(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i+1)
    return mol

def get_init_maps(mol):
    maps = []
    for i,atom in enumerate(mol.GetAtoms()):
        num = atom.GetAtomMapNum()
        if num == 0:
            maps.append(i+1)
        else:
            maps.append(num)
    return maps

def reset_smiles(rxn_smiles, shuffle='product', sanitize=True):
    print('rxn smiles', rxn_smiles)
    reactant_smiles, product_smiles = rxn_smiles.split('>>')

    if shuffle == 'reactant':
        reactant_mol = Chem.MolFromSmiles(reactant_smiles)
        reactant_mol = Chem.AddHs(reactant_mol)
        reactant_mol = set_random_atom_map(reactant_mol)
        reactant_smiles = Chem.MolToSmiles(reactant_mol)

    if shuffle == 'product':
        product_mol = Chem.MolFromSmiles(product_smiles)
        #product_mol = Chem.AddHs(product_mol)
        product_mol = set_random_atom_map(product_mol)
        product_smiles = Chem.MolToSmiles(product_mol)

    if shuffle == 'both':
        reactant_mol = Chem.MolFromSmiles(reactant_smiles, sanitize=sanitize)
        reactant_mol = Chem.AddHs(reactant_mol)
        reactant_mol = set_atom_map(reactant_mol)
        reactant_smiles = Chem.MolToSmiles(reactant_mol)
        product_mol = Chem.MolFromSmiles(product_smiles, sanitize=sanitize)
        # product_mol = Chem.AddHs(product_mol)
        product_mol = set_random_atom_map(product_mol)
        product_smiles = Chem.MolToSmiles(product_mol)

    mod_rxn_smiles = reactant_smiles + '>>' + product_smiles
    print('mod rxn smiles', mod_rxn_smiles)
    print('\n')
    return mod_rxn_smiles

if __name__ == "__main__":
    args = argparse()
    cyclo = args.cyclo
    gdb = args.gdb
    proparg = args.proparg
    proparg_good = args.proparg_good

    if cyclo:
        cyclo_df = pd.read_csv('data/cyclo/full_dataset.csv', index_col=0)
        rxn_smiles = cyclo_df['rxn_smiles']
        mod_rxn_smiles = [reset_smiles(x, shuffle='product') for x in rxn_smiles]
        cyclo_df["rxn_smiles"] = mod_rxn_smiles
        cyclo_df.to_csv("data/cyclo/random_mapped_rxns.csv")
        job_df = cyclo_df[['rxn_smiles', 'G_act']]
        job_df.to_csv("data/cyclo/submit_random_rxns.csv", index=False)

        print("File for cyclo atom maps saved")

    if gdb:
        gdb_df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv", index_col=0)
        rxn_smiles = gdb_df['rxn_smiles']
        mod_rxn_smiles = [reset_smiles(x, shuffle='reactant') for x in rxn_smiles]
        gdb_df["rxn_smiles"] = mod_rxn_smiles
        gdb_df.to_csv("data/gdb7-22-ts/random_mapped_rxns.csv")
        job_df = gdb_df[['rxn_smiles', 'dE0']]
        job_df.to_csv("data/gdb7-22-ts/submit_random_rxns.csv", index=False)

        print("File for gdb atom maps saved")

    if proparg:
        proparg_df = pd.read_csv("data/proparg/data.csv", index_col=0)
        rxn_smiles = proparg_df['rxn_smiles']
        mod_rxn_smiles = [reset_smiles(x, shuffle='both') for x in rxn_smiles]
        proparg_df["rxn_smiles"] = mod_rxn_smiles
        proparg_df.to_csv("data/proparg/random_mapped_rxns.csv")
        job_df = proparg_df[['rxn_smiles', 'Eafw']]
        job_df.to_csv("data/proparg/submit_random_rxns.csv", index=False)

        print("File for proparg atom maps saved")

    if proparg_good:
        proparg_df = pd.read_csv("data/proparg/data_good-smiles_mapped.csv", index_col=0)
        rxn_smiles = proparg_df['rxn_smiles_mapped']
        mod_rxn_smiles = [reset_smiles(x, shuffle='both', sanitize=False) for x in rxn_smiles]
        proparg_df["rxn_smiles_random"] = mod_rxn_smiles
        proparg_df.to_csv("data/proparg/data_good-smiles_mapped.csv")
        #job_df = proparg_df[['rxn_smiles_random', 'Eafw']]
        #job_df.to_csv("data/proparg/submit_random_rxns_new.csv", index=False)

        print("File for proparg atom maps saved")

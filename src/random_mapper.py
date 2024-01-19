import argparse as ap
import numpy as np
import pandas as pd
from rdkit import Chem


def argparse():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    parser.add_argument('-p', '--proparg', action='store_true')
    parser.add_argument('--proparg_arom', action='store_true')
    parser.add_argument('--proparg_stereo', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    return args


def set_random_atom_map(mol):
    atoms   = np.array([at.GetSymbol() for at in mol.GetAtoms()])
    mapping = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
    for q in sorted(set(atoms)):
        idx = np.where(atoms==q)
        maps_q = mapping[idx]
        np.random.shuffle(maps_q)
        mapping[idx] = maps_q
    for m, atom in zip(mapping, mol.GetAtoms()):
        atom.SetAtomMapNum(int(m))


def reset_smiles(rxn_smiles, shuffle='product'):
    reactant_smiles, product_smiles = rxn_smiles.split('>>')

    if shuffle == 'reactant':
        reactant_mol = Chem.MolFromSmiles(reactant_smiles, sanitize=False)
        set_random_atom_map(reactant_mol)
        reactant_smiles = Chem.MolToSmiles(reactant_mol)

    if shuffle == 'product':
        product_mol = Chem.MolFromSmiles(product_smiles, sanitize=False)
        set_random_atom_map(product_mol)
        product_smiles = Chem.MolToSmiles(product_mol)

    mod_rxn_smiles = reactant_smiles + '>>' + product_smiles
    return mod_rxn_smiles


if __name__ == "__main__":

    args = argparse()

    datasets = (
        (args.cyclo,          "data/cyclo/full_dataset.csv",                 'rxn_smiles',        'product'),
        (args.gdb,            "data/gdb7-22-ts/ccsdtf12_dz.csv",             'rxn_smiles',        'reactant'),
        (args.proparg,        "data/proparg/data.csv",                       'rxn_smiles_mapped', 'product'),
        (args.proparg_arom,   "data/proparg/data_fixarom_smiles.csv",        'rxn_smiles_mapped', 'product'),
        (args.proparg_stereo, "data/proparg/data_fixarom_smiles_stereo.csv", 'rxn_smiles_mapped', 'product'),
    )

    for flag, dfile, src_column, component in datasets:
        if flag:
            np.random.seed(args.seed)
            df = pd.read_csv(dfile, index_col=0)
            rxn_smiles = df[src_column]
            mod_rxn_smiles = [reset_smiles(x, shuffle=component) for x in rxn_smiles]
            df["rxn_smiles_random"] = mod_rxn_smiles
            df.to_csv(dfile)
            print(f"Random atom maps overwritten in {dfile}")

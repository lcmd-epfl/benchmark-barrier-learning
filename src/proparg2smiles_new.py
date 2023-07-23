#!/usr/bin/env python3

from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import ase.io
import networkx
import networkx.algorithms.isomorphism as iso


def get_atoms_in_order(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(mol)
    atoms   = np.array([at.GetSymbol() for at in mol.GetAtoms()])
    mapping = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])
    assert np.all(mapping > 0)
    atoms, mapping = map(np.array, zip(*[(at.GetSymbol(), at.GetAtomMapNum()) for at in mol.GetAtoms()]))
    return atoms[np.argsort(mapping)]


def make_nx_graph(atoms, bonds):
    G = networkx.Graph()
    G.add_nodes_from([(i, {'q': q}) for i, q in enumerate(atoms)])
    G.add_edges_from(bonds)
    return G


def make_xyz_graph(filename):

    mol = ase.io.read(filename)
    atoms = np.array(mol.get_chemical_symbols())

    cutoff = [ase.data.covalent_radii[z] for z in mol.get_atomic_numbers()]
    nl = ase.neighborlist.NeighborList(cutoff, self_interaction=False, bothways=False)
    nl.update(mol)
    AC = nl.get_connectivity_matrix(sparse=True)
    bonds = np.array(sorted(sorted(i) for i in AC.keys()))

    return make_nx_graph(atoms, bonds), atoms


def make_rdkit_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    atoms = np.array([at.GetSymbol() for at in mol.GetAtoms()])
    bonds = np.array(sorted(sorted((i.GetBeginAtomIdx(), i.GetEndAtomIdx())) for i in mol.GetBonds()))
    return make_nx_graph(atoms, bonds), atoms, mol


def map_smiles(xyz, smi1, smi2, normal_Si_coordnum):

    # Make graphs.
    # Choose SMILES from normal (smi1) and alternative (smi2)
    # based on Silicon coordination number
    G2, xyz_atoms = make_xyz_graph(xyz)
    Si_coordnum = len(G2.edges(np.where(xyz_atoms=='Si')[0][0]))
    smi = smi1 if Si_coordnum==normal_Si_coordnum else smi2
    G1, rdkit_atoms, mol = make_rdkit_graph(smi)

    # Match graphs
    GM = iso.GraphMatcher(G1, G2, node_match=iso.categorical_node_match('q', None))
    assert GM.is_isomorphic(), f'not isomorfic {xyz}'
    match = next(GM.match())
    src, dst = np.array(sorted(match.items(), key=lambda match: match[0])).T
    assert np.all(src==np.arange(G1.number_of_nodes())), xyz
    assert np.all(rdkit_atoms==xyz_atoms[dst]), xyz

    # Map atoms
    for i, at in enumerate(mol.GetAtoms()):
        at.SetAtomMapNum(int(dst[i])+1)

    # Check mapping
    maps = np.array([at.GetAtomMapNum() for at in mol.GetAtoms()])-1
    assert np.all(rdkit_atoms==xyz_atoms[maps]), xyz

    return Chem.MolToSmiles(mol)


def clean_smiles(x):
    return Chem.MolToSmiles(Chem.MolFromSmiles(x))


def construct_ligands():

    ''' Labeled Hydrodens are fragment linkers.
        1001 : bipy bond
        1002 : R bond
        1003 : Y bond
        1004 : X bond
    '''

    frags = {
        'Me'  : 'C([H:1002])'         ,
        'Ph'  : 'c1([H:1003])ccccc1'  ,
        'tBu' : 'CC(C)(C)([H:1003])'  ,
    }

    X_frags = {
        'a' : '[H]([H:1004])'        ,  # H
        'b' : '[F]([H:1004])'        ,  # F
        'c' : '[Cl]([H:1004])'       ,  # Cl
        'd' : 'C([H:1004])'          ,  # CH3
        'e' : 'C(F)(F)(F)([H:1004])' ,  # CF3
        'f' : 'CC(C)([H:1004])'      ,  # iPr
        'g' : 'CC(C)(C)([H:1004])'   ,  # tBu
        'h' : 'C#C([H:1004])'        ,  # CCH
        'i' : 'C([H:1004])#N'        ,  # CN
        'j' : 'c1([H:1004])ccccc1'   ,  # Ph
    }

    def concatenate_fragments(a, b, nH):
        return (a+'.'+b).replace(f'([H:{nH}])', f'%{nH-1000+20}')

    scaffolds = {}
    scaffolds[1, 'bp'] = '[H]c1c([H:1002])c([H:1001])[n+]([O-])c([H:1004])c1([H:1003])'
    scaffolds[2, 'bp'] = concatenate_fragments(scaffolds[1, 'bp'], frags['Ph'],  1003)
    scaffolds[3, 'bp'] = concatenate_fragments(scaffolds[1, 'bp'], frags['tBu'], 1003)
    scaffolds[4, 'bp'] = '[H]c1c([H])c([H:1004])c2c(c1[H])c([H])c([H:1002])c([H:1001])[n+]2[O-]'
    scaffolds[5, 'bp'] = '[H]c1c([H])c([H])c2c([H:1004])[n+]([O-])c([H:1001])c([H:1002])c2c1[H]'
    scaffolds[6, 'bp'] = '[H]c1c([H])c([H])c2c([H:1001])[n+]([O-])c([H:1004])c([H])c2c1[H]'

    for i in range(1, 7):
        if '1002' in scaffolds[i, 'bp']:
            scaffolds[i, 'Mebp'] = concatenate_fragments(scaffolds[i, 'bp'], frags['Me'], 1002)

    scaffolds_3aMebp = concatenate_fragments(scaffolds[1, 'bp'], frags['Me'].replace('1002','1003'), 1003)
    scaffolds_3aMebp = concatenate_fragments(scaffolds_3aMebp, frags['Me'], 1002)

    ligands = {}
    for (Y, R), scaffold in scaffolds.items():
        for X, X_frag in X_frags.items():
            key = f'{Y}{X}{R}'
            if key=='3aMebp':
                tmp = concatenate_fragments(scaffolds_3aMebp, X_frag, 1004)
            else:
                tmp = concatenate_fragments(scaffold, X_frag, 1004)
            tmp = concatenate_fragments(tmp, tmp, 1001)
            tmp = clean_smiles(tmp)
            ligands[key] = tmp

    return ligands


def construct_reactants_products(ligands):

    reactions = {}

    for key, ligand in ligands.items():

        lig_1dent = ligand.replace('[O-]', 'O%31', 1)
        lig = lig_1dent.replace('[O-]', 'O%32', 1)

        center_product = '[Si+](Cl)(Cl)%31%32%33'
        product = 'C(#CCC(C1=CC=CC=C1)O%33)[H]'
        complex_product = lig+'.'+center_product+'.'+product

        center_reactant = '[Si](Cl)(Cl)%31%32%33%34'
        reactant1 = 'C=C=C%33'
        # TODO ask Simone
        reactant2 = 'C1(=CC=CC=C1)C=[O+]%34' if True else 'C1(=CC=CC=C1)[CH+][O]%34'
        complex_reactant = lig+'.'+ center_reactant+'.'+reactant1+'.'+reactant2

        # TODO ask Simone
        center_reactant_5val = '[Si+](Cl)(Cl)%31%33%34'
        complex_reactant_1SiONbond = lig_1dent+'.'+center_reactant_5val+'.'+reactant1+'.'+reactant2

        # TODO ask Simone
        center_product_4val = '[Si](Cl)(Cl)%31%33'
        complex_product_1SiONbond = lig_1dent+'.'+center_product_4val+'.'+product

        reactions[key] = SimpleNamespace(reactant = clean_smiles(complex_reactant),
                                         product  = clean_smiles(complex_product),
                                         reactant_1SiONbond = clean_smiles(complex_reactant_1SiONbond),
                                         product_1SiONbond  = clean_smiles(complex_product_1SiONbond))
    return reactions


def main(data_dir='data/proparg'):

    ligands = construct_ligands()
    reactions = construct_reactants_products(ligands)

    df = pd.read_csv(f'{data_dir}/data.csv', index_col=0)
    labels = df['mol'].values
    enans  = df['enan'].values

    for label, enan in tqdm(zip(labels, enans), total=len(df)):

        xyz_reactant, xyz_product = [f'{data_dir}/data_{comp}_xyz/{label}{enan}.xyz' for comp in ('react', 'prod')]
        smis = reactions[label[:-1]]

        try:
            reactant_mapped = map_smiles(xyz_reactant, smis.reactant, smis.reactant_1SiONbond, 6)
            product_mapped  = map_smiles(xyz_product,  smis.product,  smis.product_1SiONbond,  5)
            arm, apm = map(get_atoms_in_order, (reactant_mapped, product_mapped))
            assert np.all(arm==apm)

        except Exception as e:
            print('\n', '\033[1;91m', e, '\033[0m')
            continue

        df.loc[(df['mol']==label) & (df['enan']==enan), 'rxn_smiles_mapped'] = reactant_mapped+'>>'+product_mapped
    df.to_csv('proparg-mapped.csv')

if __name__=='__main__':
    main()

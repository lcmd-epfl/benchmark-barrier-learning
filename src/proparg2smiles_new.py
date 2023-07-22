#!/usr/bin/env python3

from rdkit import Chem


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

    ligands = {}
    for (Y, R), scaffold in scaffolds.items():
        for X, X_frag in X_frags.items():
            tmp = concatenate_fragments(scaffold, X_frag, 1004)
            tmp = concatenate_fragments(tmp, tmp, 1001)
            tmp = clean_smiles(tmp)
            ligands[f'{Y}{X}{R}'] = tmp

    return ligands


def construct_reactants_products(ligands):

    products = {}
    reactants = {}

    for key, lig in ligands.items():

        lig = lig.replace('[O-]', 'O%31', 1).replace('[O-]', 'O%32', 1)

        center_product = '[Si+](Cl)(Cl)%31%32%33'
        product = 'C(#CCC(C1=CC=CC=C1)O%33)[H]'
        complex_product = lig+'.'+center_product+'.'+product
        complex_product = clean_smiles(complex_product)
        products[key] = complex_product

        center_reactant = '[Si](Cl)(Cl)%31%32%33%34'
        reactant1 = 'C=C=C%33'
        reactant2 = 'C1(=CC=CC=C1)C=[O+]%34' if True else 'C1(=CC=CC=C1)[CH+][O]%34'   # TODO ask Simone
        complex_reactant = lig+'.'+ center_reactant+'.'+reactant1+'.'+reactant2
        complex_reactant = clean_smiles(complex_reactant)
        reactants[key] = complex_reactant

    return products, reactants


def main(data_dir='benchmark-barrier-learning/data/proparg'):

    ligands = construct_ligands()
    products, reactants = construct_reactants_products(ligands)

    for key in ligands.keys():
        print(key)
        print(products[key])
        print(reactants[key])
        print()


if __name__=='__main__':
    main()

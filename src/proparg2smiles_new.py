#!/usr/bin/env python3

from types import SimpleNamespace
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


def main(data_dir='benchmark-barrier-learning/data/proparg'):

    ligands = construct_ligands()
    reactions = construct_reactants_products(ligands)

    for key in ligands.keys():
        print(key)
        print(reactions[key].reactant)
        print(reactions[key].reactant_1SiONbond)
        print(reactions[key].product)
        print(reactions[key].product_1SiONbond)
        print()


if __name__=='__main__':
    main()

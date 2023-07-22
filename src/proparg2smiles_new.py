#!/usr/bin/env python3

from rdkit import Chem


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

    def concatenate_fragments(a, b, nH, nbond=9):
        return (a + '.' + b).replace(f'([H:{nH}])', f'%{nbond+nH-1000+10}')

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
            tmp = Chem.MolToSmiles(Chem.MolFromSmiles(tmp))
            ligands[f'{Y}{X}{R}'] = tmp

    return ligands


def main(data_dir='benchmark-barrier-learning/data/proparg'):
    ligands = construct_ligands()
    for i, j in ligands.items():
        print(i, j)


if __name__=='__main__':
    main()

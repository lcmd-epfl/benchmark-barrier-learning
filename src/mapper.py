#!/usr/bin/env python3

import argparse as ap
from operator import itemgetter
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper

# rxnmapper==0.3.0


def argparse():
    parser = ap.ArgumentParser()
    parser.add_argument('-c', '--cyclo', action='store_true')
    parser.add_argument('-g', '--gdb', action='store_true')
    parser.add_argument('-p', '--proparg', action='store_true')
    parser.add_argument('--proparg_arom', action='store_true')
    parser.add_argument('--proparg_stereo', action='store_true')
    args = parser.parse_args()
    return args


def reset_smiles(x):
    return re.sub(':[0-9]+', '', x)


if __name__ == "__main__":

    args = argparse()
    datasets = (
        (args.cyclo,        "data/cyclo/full_dataset.csv",          'rxn_smiles'       ),
        (args.gdb,          "data/gdb7-22-ts/ccsdtf12_dz.csv",      'rxn_smiles'       ),
        (args.proparg,      "data/proparg/data.csv",                'rxn_smiles_mapped'),
    )

    for flag, dfile, src_column in datasets:
        if flag:
            data = pd.read_csv(dfile, index_col=0)
            reactions = data[src_column].to_list()
            reactions = [*map(reset_smiles, reactions)]

            rxn_mapper = RXNMapper()
            results = [rxn_mapper.get_attention_guided_atom_maps([r])[0] for r in tqdm(reactions)]
            reactions, confidence = zip(*map(itemgetter('mapped_rxn', 'confidence'), results))

            print(np.mean(list(confidence)))
            data['rxn_smiles_rxnmapper'] = list(reactions)
            data.to_csv(dfile)
            print(f"rxnmapper atom maps overwritten in {dfile}")

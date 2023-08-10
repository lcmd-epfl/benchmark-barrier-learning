#!/usr/bin/env python3

import argparse as ap
import chemprop

# chemprop==1.5.0
#
# to make chemprop work with data_fixarom_smiles.csv,
# patch ${CONDA_PREFIX}/lib/python3.8/site-packages/chemprop/rdkit.py
# with cgr_proparg_patch.txt


def argparse():
    parser = ap.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group()
    g1.add_argument('--random', action='store_true')
    g1.add_argument('--rxnmapper', action='store_true')
    g2 = parser.add_mutually_exclusive_group()
    g2.add_argument('--cyclo', action='store_true')
    g2.add_argument('--gdb_full', action='store_true')
    g2.add_argument('--gdb_mod', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = argparse()
    if args.cyclo:
        data_path = '../../data/cyclo/full_dataset.csv'
        target_columns = 'G_act'
    elif args.gdb_full:
        data_path = '../../data/gdb7-22-ts/ccsdtf12_dz.csv'
        target_columns = 'dE0'
    elif args.gdb_mod:
        data_path = '../../data/gdb7-22-ts/ccsdtf12_dz_mod.csv'
        target_columns = 'dE0'
    else:
        exit(0)
    if args.random:
        smiles_columns = 'rxn_smiles_random'
    elif args.rxnmapper:
        smiles_columns = 'rxn_smiles_rxnmapper'
    else:
        smiles_columns = 'rxn_smiles'


    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--metric", "mae",
        "--dropout", "0.05",
        "--epochs", "300",
        "--reaction",
        "--num_folds",  "10",
        "--batch_size", "50",
        "--save_dir", "./"]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    print("Mean score", mean_score, "std_score", std_score)

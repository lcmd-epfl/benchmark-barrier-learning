#!/usr/bin/env python3

import sys
import argparse as ap
import chemprop

def argparse():
    parser = ap.ArgumentParser()
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument('-c', '--cyclo', action='store_true', help='use Cyclo-23-TS dataset')
    g2.add_argument('-g', '--gdb', action='store_true', help='use GDB7-22-TS dataset')
    g2.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset with SMILES from xyz')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":
    dataset = ''
    parser, args = argparse()
    if args.cyclo:
        data_path = '../data/cyclo/full_dataset.csv'
        target_columns = 'G_act'
        dataset = 'cyclo'
    elif args.gdb:
        data_path = '../data/gdb7-22-ts/ccsdtf12_dz.csv'
        target_columns = 'dE0'
        dataset = 'gdb'
    elif args.proparg:
        data_path = '../data/proparg/data.csv'
        target_columns = "Eafw"
        dataset = 'proparg'

    # use true mapping for hypers 
    if args.proparg:
        smiles_columns = 'rxn_smiles_mapped'
    else:
        smiles_columns = 'rxn_smiles'

    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--metric", "mae",
        "--epochs", "100",
        "--num_iters", "100",
        "--config_save_path", f"hypers_{dataset}.json",
        "--reaction",
        "--num_folds",  "10",
        "--batch_size", "50",
        '--explicit_h']

    args_chemprop = chemprop.args.HyperoptArgs().parse_args(arguments)
    chemprop.hyperparameter_optimization.hyperopt(args=args_chemprop)
    print("File saved")

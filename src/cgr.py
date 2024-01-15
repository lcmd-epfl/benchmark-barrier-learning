#!/usr/bin/env python3

import sys
import argparse as ap
import chemprop

# to make chemprop work with --proparg_combinat/--proparg_stereo,
# patch ${CONDA_PREFIX}/lib/python3.8/site-packages/chemprop/rdkit.py
# with cgr_proparg_patch.txt


def argparse():
    parser = ap.ArgumentParser()
    g1 = parser.add_mutually_exclusive_group(required=True)
    g1.add_argument('--true', action='store_true', help='use true atom mapping')
    g1.add_argument('--rxnmapper', action='store_true', help='use atom mapping from rxnmapper')
    g1.add_argument('--nomapping', action='store_true', help='use without atom mapping')
    g2 = parser.add_mutually_exclusive_group(required=True)
    g2.add_argument('-c', '--cyclo', action='store_true', help='use Cyclo-23-TS dataset')
    g2.add_argument('-g', '--gdb', action='store_true', help='use GDB7-22-TS dataset')
    g2.add_argument('-p', '--proparg', action='store_true', help='use Proparg-21-TS dataset with SMILES from xyz')
    g2.add_argument('--proparg_combinat', action='store_true', help='use Proparg-21-TS dataset with fragment-based SMILES')
    g2.add_argument('--proparg_stereo', action='store_true', help='use Proparg-21-TS dataset with stereochemistry-enriched fragment-based SMILES')
    parser.add_argument('--withH', action='store_true', help='use explicit H')
    args = parser.parse_args()
    return parser, args


if __name__ == "__main__":

    parser, args = argparse()
    if args.cyclo:
        data_path = '../../data/cyclo/full_dataset.csv'
        target_columns = 'G_act'
    elif args.gdb:
        data_path = '../../data/gdb7-22-ts/ccsdtf12_dz.csv'
        target_columns = 'dE0'
    elif args.proparg:
        data_path = '../../data/proparg/data.csv'
        target_columns = "Eafw"
    elif args.proparg_combinat:
        data_path = '../../data/proparg/data_fixarom_smiles.csv'
        target_columns = "Eafw"
    elif args.proparg_stereo:
        data_path = '../../data/proparg/data_fixarom_smiles_stereo.csv'
        target_columns = "Eafw"

    if args.rxnmapper:
        smiles_columns = 'rxn_smiles_rxnmapper'
    elif args.nomapping:
        if args.proparg or args.proparg_combinat or args.proparg_stereo:
            smiles_columns = 'rxn_smiles'
        else:
            smiles_columns = 'rxn_smiles_nomapping'
    elif args.true:
        if args.proparg or args.proparg_combinat or args.proparg_stereo:
            smiles_columns = 'rxn_smiles_mapped'
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
    if args.withH:
        arguments.append('--explicit_h')

    args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
    print("Mean score", mean_score, "std_score", std_score)

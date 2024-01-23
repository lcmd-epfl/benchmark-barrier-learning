#!/usr/bin/env python3

import sys
import argparse as ap
import chemprop

# to make chemprop work with --proparg_combinat/--proparg_stereo,
# patch ${CONDA_PREFIX}/lib/python{version}/site-packages/chemprop/rdkit.py
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
    dataset = ''
    dataset_spec = ''
    if args.cyclo:
        data_path = '../data/cyclo/full_dataset.csv'
        target_columns = 'G_act'
        dataset = 'cyclo'
        dataset_spec = 'cyclo'
    elif args.gdb:
        data_path = '../data/gdb7-22-ts/ccsdtf12_dz.csv'
        target_columns = 'dE0'
        dataset = 'gdb'
        dataset_spec = 'gdb'
    elif args.proparg:
        data_path = '../data/proparg/data.csv'
        target_columns = "Eafw"
        dataset = 'proparg'
        dataset_spec = 'proparg'
    elif args.proparg_combinat:
        data_path = '../data/proparg/data_fixarom_smiles.csv'
        target_columns = "Eafw"
        dataset = 'proparg'
        dataset_spec = 'proparg_combinat'
    elif args.proparg_stereo:
        data_path = '../data/proparg/data_fixarom_smiles_stereo.csv'
        target_columns = "Eafw"
        dataset = 'proparg'
        dataset_spec = 'proparg_stereo'
    config_path = f"../data/hypers_{dataset}_cgr.json"

    results_dir = ''
    results_base_dir = f"../results/{dataset_spec}_"
    if args.rxnmapper:
        smiles_columns = 'rxn_smiles_rxnmapper'
        results_dir = results_base_dir + "rxnmapper" 
    elif args.nomapping:
        results_dir = results_base_dir + "nomap"
        if args.proparg or args.proparg_combinat or args.proparg_stereo:
            smiles_columns = 'rxn_smiles'
        else:
            smiles_columns = 'rxn_smiles_nomapping'
    elif args.true:
        results_dir = results_base_dir + "true"
        if args.proparg or args.proparg_combinat or args.proparg_stereo:
            smiles_columns = 'rxn_smiles_mapped'
        else:
            smiles_columns = 'rxn_smiles'

    if args.withH:
        results_dir += '_withH'

    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--metric", "mae",
        "--epochs", "300",
        "--reaction",
        "--config_path", config_path,
        "--num_folds",  "10",
        "--batch_size", "50",
        "--save_dir", results_dir]
    if args.withH:
        arguments.append('--explicit_h')

    args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
    print("Mean score", mean_score, "std_score", std_score)

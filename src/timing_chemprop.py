#!/usr/bin/env python3

import sys
import chemprop

if __name__ == "__main__":

    dataset = ['proparg', 'cyclo', 'gdb']
    subset= 750

    for dataset in datasets:
        print(f"{dataset=}")
        config_path = f"../../data/hypers_{dataset}_cgr.json"

        if dataset == 'cyclo':
            data_path = '../../data/cyclo/full_dataset.csv'
            target_columns = 'G_act'
            smiles_columns = 'rxn_smiles'

        elif dataset == 'gdb':
            data_path = '../../data/gdb7-22-ts/ccsdtf12_dz.csv'
            target_columns = 'dE0'
            smiles_columns = 'rxn_smiles'
        elif dataset == 'proparg':
            data_path = '../../data/proparg/data.csv'
            target_columns = "Eafw"
            smiles_columns = 'rxn_smiles_mapped'

    arguments = [
        "--data_path", data_path,
        "--dataset_type",  "regression",
        "--target_columns", target_columns,
        "--smiles_columns", smiles_columns,
        "--metric", "mae",
        "--epochs", "300",
        "--reaction",
        "--config_path", config_path,
        "--num_folds",  "1",
        "--batch_size", "50",
        "--save_dir", results_dir]
    if dataset != 'cyclo':
        arguments.append('--explicit_h')

    args_chemprop = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args_chemprop, train_func=chemprop.train.run_training)
    # timing will be dumped in out file

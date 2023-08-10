#!/usr/bin/env python3

import argparse as ap
import chemprop

# chemprop==1.5.0
#
# to make chemprop work with data_fixarom_smiles.csv,
# patch ${CONDA_PREFIX}/lib/python3.8/site-packages/chemprop/rdkit.py
# with cgr_proparg_patch.txt

parser = ap.ArgumentParser()
parser.add_argument('--smiles', default=None, type=str, help='xyz2mol/combinat/stereo')
parser.add_argument('--mapping', default=None, type=str, help='true/rxnmapper/random')
args = parser.parse_args()

if args.smiles=='xyz2mol':
    data_path = '../../data/proparg/data.csv'
elif args.smiles=='combinat':
    data_path = '../../data/proparg/data_fixarom_smiles.csv'
elif args.smiles=='stereo':
    data_path = '../../data/proparg/data_fixarom_smiles_stereo.csv'
else:
    exit()
if args.mapping=='true':
    smiles_columns = 'rxn_smiles_mapped'
elif args.mapping=='rxnmapper':
    smiles_columns = 'rxn_smiles_rxnmapper'
elif args.mapping=='random':
    smiles_columns = 'rxn_smiles_random'
else:
    exit()

arguments = [
    "--data_path", data_path,
	"--dataset_type",  "regression",
	"--target_columns", "Eafw",
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

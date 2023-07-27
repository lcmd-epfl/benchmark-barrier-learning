import chemprop

# chemprop==1.5.0
#
# to make chemprop work with data_fixarom_smiles.csv,
# patch ${CONDA_PREFIX}/lib/python3.8/site-packages/chemprop/rdkit.py
# with cgr_proparg_patch.txt

data = 'bad rxnmapper'

if data=='bad random':
    data_path = '../../data/proparg/data.csv'; smiles_columns = 'rxn_smiles_random'
elif data=='bad mapped':
    data_path = '../../data/proparg/data.csv'; smiles_columns = 'rxn_smiles_mapped'
elif data=='bad rxnmapper':
    data_path = '../../data/proparg/data.csv'; smiles_columns = 'rxn_smiles_rxnmapper'
elif data=='good mapped':
    data_path = '../../data/proparg/data_fixarom_smiles.csv'; smiles_columns = 'rxn_smiles_mapped'
elif data=='good random':
    data_path = '../../data/proparg/data_fixarom_smiles.csv'; smiles_columns = 'rxn_smiles_random'
else:
    raise NotImplementedError

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

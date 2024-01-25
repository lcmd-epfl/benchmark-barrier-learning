from rxnfp.tokenization import get_default_tokenizer, SmilesTokenizer
from rdkit import Chem
import pandas as pd
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
from transformers import BertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
import numpy as np
import argparse as ap
import glob
import random
import hypers
from timeit import default_timer as timer

def remove_atom_mapping(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            print('cannot remove atom mapping from smi')
            return smi
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        mol = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    else:
        mol = m

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    smiles = Chem.MolToSmiles(mol)

    return smiles

def remove_atom_mapping_rxn(rxn_smiles):
    r, p = rxn_smiles.split('>>')
    r_smi = remove_atom_mapping(r)
    p_smi = remove_atom_mapping(p)
    return r_smi+'>>'+p_smi

def randomize_smiles(smiles, random_type="rotated", isomericSmiles=True):
    """
    From: https://github.com/undeadpixel/reinvent-randomized and https://github.com/GLambard/SMILES-X
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted, rotated) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    if random_type == "unrestricted":
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=isomericSmiles)
    elif random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=isomericSmiles)
    elif random_type == 'rotated':
        n_atoms = mol.GetNumAtoms()
        rotation_index = random.randint(0, n_atoms-1)
        atoms = list(range(n_atoms))
        new_atoms_order = (atoms[rotation_index%len(atoms):]+atoms[:rotation_index%len(atoms)])
        rotated_mol = Chem.RenumberAtoms(mol,new_atoms_order)
        return Chem.MolToSmiles(rotated_mol, canonical=False, isomericSmiles=isomericSmiles)
    raise ValueError("Type '{}' is not valid".format(random_type))

def randomize_rxn(rxn, random_type):
    """
    Split reaction into precursors and products, then randomize all molecules.
    """
    precursors, product = rxn.split('>>')
    precursors_list = precursors.split('.')

    randomized_precursors = [randomize_smiles(precursor, random_type) for precursor in precursors_list]
    randomized_product = randomize_smiles(product, random_type)
    return f"{'.'.join(randomized_precursors)}>>{randomized_product}"


def do_randomizations_on_df(df, n_randomizations=1, random_type='rotated', seed=42):
    """
    Randomize all molecule SMILES of the reactions in a dataframe.
    Expected to have column 'text' with the reactions and 'label' with the property to predict.
    """
    new_texts = []
    new_labels = []
    random.seed(seed)

    for i, row in df.iterrows():
        if random_type != '':
            randomized_rxns = [randomize_rxn(row['text'], random_type=random_type) for i in range(n_randomizations)]
        new_texts.extend(randomized_rxns)
        new_labels.extend([row['labels']] * len(randomized_rxns))
    return pd.DataFrame({'text': new_texts, 'labels': new_labels})

def get_data(dataset):
    if dataset == 'gdb':
        df = pd.read_csv("../data/gdb7-22-ts/ccsdtf12_dz_subset_750.csv")
        target_label = 'dE0'
        save_path = 'gdb_subset'

    elif dataset == 'cyclo':
        df = pd.read_csv("../data/cyclo/dataset_subset_750.csv")
        target_label = 'G_act'
        save_path = 'cyclo_subset'

    elif dataset == 'proparg':
        target_label = 'Eafw'
        df = pd.read_csv("../data/proparg/data_subset_750.csv")
        save_path = 'proparg_subset'
    return df, target_label, save_path

def hypers_in_file():
    if hypers.HYPERS_LANG[dataset]:
        lr = hypers.HYPERS_LANG[dataset]['lr']
        p = hypers.HYPERS_LANG[dataset]['p']
        print(f"Using best hypers from file {lr} and {p}")
        return lr, p
    else:
        return False

if __name__ == "__main__":
    args = argparse()
    for dataset in ['cyclo', 'gdb', 'proparg']:
        CV = 1
        train_size = 0.8
        num_train_epochs = 5
        if dataset == 'proparg':
            batch_size = 8
        else:
            batch_size = 32
        n_randomizations = 10

        if n_randomizations > 0:
            data_aug = True
        else:
            data_aug = False

        print("Using dataset", dataset)
        df, target_label, save_path = get_data(dataset)

        rxn_smiles_list = [remove_atom_mapping_rxn(x) for x in df['rxn_smiles'].to_list()]
        df['clean_rxn_smiles'] = rxn_smiles_list
        smiles_tokenizer = get_default_tokenizer()
        df = df[['clean_rxn_smiles', target_label]]
        df.columns = ['text', 'labels']
        seed = 0

        hypers_bool = hypers_in_file()

        wandb_name_orig = str(num_train_epochs) + '_epochs_' + str(batch_size) + '_batches_' + str(n_randomizations) + '_smiles_rand'

        save_path = save_path + '/' + wandb_name_orig

        maes = []
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, SmilesTokenizer),
        }

        model_path = pkg_resources.resource_filename(
            "rxnfp",
            f"models/transformers/bert_pretrained"  # change pretrained to ft to start from the other base model
        )

        for i in range(CV):

            if CV > 1:
                wandb_name = wandb_name_orig + '.cv' + str(i+1)
            else:
                wandb_name = wandb_name_orig

            print("CV iter", i+1, '/', CV)
            save_iter_path = save_path + f"/split_{i+1}"
            seed += 1

            train_df, val_test_df = train_test_split(df, train_size=train_size, random_state=seed)
            val_df, test_df = train_test_split(df, train_size=0.5, shuffle=False)
            mean = train_df['labels'].mean()
            std = train_df['labels'].std()
            train_df['labels'] = (train_df['labels'] - mean)/std
            test_df['labels'] = (test_df['labels'] - mean)/std
            val_df['labels'] = (val_df['labels'] - mean)/std

            if data_aug:
                # now augmentation
                start_time = timer()
                train_df = do_randomizations_on_df(train_df, n_randomizations=n_randomizations, random_type='rotated', seed=seed)
                val_df = do_randomizations_on_df(val_df, n_randomizations=n_randomizations, random_type='rotated', seed=seed)
                end_time = timer()
                print(f"Train data augmentation time elapsed {end_time - start_time}")
                test_df = do_randomizations_on_df(test_df, n_randomizations=n_randomizations, random_type='rotated', seed=seed)
                end_end_time = timer()
                print(f"Test data augmentation time elapsed {end_end_time - end_time}")

            if i == 0:
                if not hypers_bool:
                    lr, p = hyperopt(wandb_name, save_iter_path, model_path, train_df, val_df, mean, std)
                else:
                    lr, p = hypers_bool


            wandb_name_search = wandb_name + f'_lr_{lr}_p_{p}'
            model_args = {'regression': True, 'evaluate_during_training': False, 'num_labels': 1,
                        'manual_seed': 2,
                        'num_train_epochs': num_train_epochs, 'wandb_project': 'lang-rxn',
                        'train_batch_size': batch_size,
                        'wandb_kwargs': {'name': wandb_name_search},
                        'learning_rate':lr,
                        "config": {'hidden_dropout_prob': p}
                        }
            path_to_save = save_iter_path+f'_lr_{lr}_p_{p}'
            path_to_search = path_to_save+f'/checkpoint*{num_train_epochs}/'

            path = glob.glob(path_to_search, recursive=True)
            if i == 0:
                if len(path) == 0:
                    print(f"training model...")
                    start_time = timer()
                    bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args,
                                                    use_cuda=torch.cuda.is_available())

                    bert.train_model(train_df, output_dir=path_to_save, eval_df=val_df)
                    end_time = timer()
                    print(f"Train time elapsed {end_time - start_time}")
                else:
                    print(f"using trained model at {path}")
            if i > 0 :
                print(f"training model...")
                start_time = timer()
                bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args,
                                                use_cuda=torch.cuda.is_available())

                bert.train_model(train_df, output_dir=path_to_save, eval_df=val_df)
                end_time = timer()
                print(f"Train time elapsed {end_time - start_time}")

            path = glob.glob(path_to_search, recursive=True)
            assert len(path) == 1, f"search path {path_to_search} contains {path}"
            model_path = path[0]
            print(f"using model path {model_path} and args {model_args}")
            start_time = timer()
            trained_bert = SmilesClassificationModel('bert', model_path,
                                                        num_labels=1, args=model_args,
                                                        use_cuda=torch.cuda.is_available())

            predictions = trained_bert.predict(test_df.text.values.tolist())[0]
            end_time = timer()
            print(f"End time elapsed {end_time - start_time}")

            predictions = predictions * std + mean

            true = test_df['labels'] * std + mean

            mae = np.mean(np.abs(true - predictions))
            maes.append(mae)

        print(f"test MAE={np.mean(maes)} +- {np.std(maes)}")
        savefile = save_path + '/results.txt'
        with open(savefile, 'w') as f:
            for mae in maes:
                f.write(str(mae))
                f.write('\n')
        print("Results file saved to", savefile)

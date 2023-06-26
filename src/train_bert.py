from rxnfp.tokenization import get_default_tokenizer, SmilesTokenizer
from rdkit import Chem
import pandas as pd
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
from sklearn.model_selection import train_test_split
import numpy as np
import argparse as ap
import glob

def argparse():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-d', '--dataset', default='gdb')
    parser.add_argument('-CV', '--CV', default=1)
    parser.add_argument('-test_size', '--test_size', default=0.2)
    args = parser.parse_args()

    args.CV = int(args.CV)
    args.test_size = float(args.test_size)
    return args

def remove_atom_mapping(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("could not convert smi", smi, "to mol")
        return smi
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

if __name__ == "__main__":
    args = argparse()
    train = args.train
    predict = args.predict
    dataset = args.dataset
    CV = args.CV
    test_size = args.test_size

    print("Using dataset", dataset)

    if dataset == 'gdb':
        df = pd.read_csv("data/gdb7-22-ts/ccsdtf12_dz.csv")
        target_label = 'dE0'
        save_path = 'outs/gdb_bert_pretrained'

    elif dataset == 'cyclo':
        df = pd.read_csv("data/cyclo/full_dataset.csv")
        target_label = 'G_act'
        save_path = 'outs/cyclo_bert_pretrained'

    elif dataset == 'proparg':
        df = pd.read_csv("data/proparg/data.csv")
        target_label = 'Eafw'
        save_path = 'outs/proparg_bert_pretrained'

    rxn_smiles_list = [remove_atom_mapping_rxn(x) for x in df['rxn_smiles'].to_list()]
    df['clean_rxn_smiles'] = rxn_smiles_list
    smiles_tokenizer = get_default_tokenizer()
    # prepare data train/test splits
    df = df[['clean_rxn_smiles', target_label]]
    df.columns = ['text', 'labels']
    seed = 0

    maes = []
    for i in range(CV):
        print("CV iter", i+1, '/', CV)
        save_iter_path = save_path + f"/split_{i+1}"
        seed += 1
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

        mean = train_df['labels'].mean()
        std = train_df['labels'].std()
        train_df['labels'] = (train_df['labels'] - mean)/std
        test_df['labels'] = (test_df['labels'] - mean)/std

        print('tr size', len(train_df), 'te size', len(test_df))

        if train:
            model_args = {'regression':True, 'evaluate_during_training':False, 'num_labels':1, 'manual_seed':2}

            model_path = pkg_resources.resource_filename(
                            "rxnfp",
                            f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
            )
            bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args,
                                                   use_cuda=torch.cuda.is_available())

            bert.train_model(train_df, output_dir=save_iter_path, eval_df=test_df)

        if predict:
            path = glob.glob(save_iter_path+'/checkpoint*/', recursive=True)
            assert len(path) == 1
            model_path = path[0]
            print(f"using model path {model_path}")
            trained_bert = SmilesClassificationModel('bert', model_path, num_labels=1, args={'regression':True},
                                                     use_cuda=torch.cuda.is_available())
            predictions = trained_bert.predict(test_df.text.values)[0]
            predictions = predictions * std + mean

            true = test_df['labels'] * std + mean

            mae = np.mean(np.abs(true - predictions))
            maes.append(mae)

    print(f"MAE={np.mean(maes)} +- {np.std(maes)}")
    savefile = save_path + '/results.txt'
    with open(savefile, 'w') as f:
        for mae in maes:
            f.write(str(mae))
            f.write('\n')
    print("Results file saved to", savefile)
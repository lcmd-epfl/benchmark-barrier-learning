from src.smilify import smilify
import pandas as pd

sn2_df = pd.read_csv("data/sn2-e2/final_sn2.csv")
smiles_reactants = []
smiles_products = []
for fname in sn2_df['lowest energy conformer filenames'].tolist():
    full_fname = 'data/sn2-e2/' + fname
    smi = smilify(full_fname)
    smiles_reactants.append(smi)

for fname in sn2_df['product_file'].tolist():
    full_fname = 'data/sn2-e2/' + fname
    smi = smilify(full_fname)
    smiles_products.append(smi)

#TODO FINISH HERE
sn2_df['SMILES reactants']

e2_df = pd.read_csv("data/sn2-e2/final_e2.csv")
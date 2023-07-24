import pandas as pd
from src.smilify import smilify

data = pd.read_csv("data/proparg/data.csv", index_col=0)
reactants_files = [
    "data/proparg/data_react_xyz/"
    + data.mol.values[i]
    + data.enan.values[i]
    + ".xyz"
    for i in range(len(data))
]
products_files = [
    "data/proparg/data_prod_xyz/"
    + data.mol.values[i]
    + data.enan.values[i]
    + ".xyz"
    for i in range(len(data))
]

reactant_smiles = []
for i, file in enumerate(reactants_files):
    smi = smilify(file)
    if smi is None:
        smi = ''
    reactant_smiles.append(smi)

product_smiles = []
for i, file in enumerate(products_files):
    smi = smilify(file)
    if smi is None:
        smi = ''
    product_smiles.append(smi)

reaction_smiles = [r+'>>'+p for (r,p) in zip(reactant_smiles, product_smiles)]
data['rxn_smiles'] = reaction_smiles
data.to_csv('data/proparg/data.csv')
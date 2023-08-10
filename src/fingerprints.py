from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem import AllChem, rdChemReactions
import rdkit.Chem as Chem
import numpy as np
from drfp import DrfpEncoder

def get_MFP(rxn_smiles):
    params = rdChemReactions.ReactionFingerprintParams()
    params.fpSize = 1024
    params.fpType = rdChemReactions.FingerprintType.MorganFP
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True, replacements={'-2':''})
    rxn_fp = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn, params)
    converted_fp = np.zeros((1))
    ConvertToNumpyArray(rxn_fp, converted_fp)
    return converted_fp

def get_DRFP(rxn_smiles):
    return DrfpEncoder.encode(rxn_smiles, n_folded_length=1024)[0]

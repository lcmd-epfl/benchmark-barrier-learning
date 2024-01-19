# Benchmarking of reaction representations

## Warning 
This repo is currently in progress, for the revision of the paper. We will remove this warning when the repo is finalised.

## Installation
- Due to the various dependencies of different packages, it is not possible to have a single environment that satisfies them all. In particular, the `rxnfp` package relies on python 3.6. Therefore it is recommended to have a separate environment to run the BERT+RXNFP models:
```commandline
conda create -n rxnfp python=3.7 -y
conda activate rxnfp
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c tmap tmap -y
git clone https://github.com/rxn4chemistry/rxnfp
pip install -e rxnfp
```
- Note that the above does not work on Mac computers with the M1 chip onwards, since they do not support python 3.7. This env has been tried and tested on several Linux platforms.
- Then another environment can be created that works for the other fingerprints, for example `conda create -n benchmark-rxn`
- `conda install pip`
- Due to the dependencies of the qml python package, numpy needs to be installed first and foremost, therefore a `requirements.txt` file is provided
- Then install requirements like `xargs -L 1 pip install < requirements.txt`

## 2D and 3D fingerprints
- 2D reps here are the DRFP and MFP
- 3D reps are SLATM and $B^2R^2_l$, all in the difference variation for reactions
- To generate/load the reps and perform CV-fold cross-validated predictions with a train fraction of tr, the file `src/run_all_fingerprints.py` should be used
- Here, the datasets are specified as additional arguments, `-c` for Cyclo-23-TS, `-g` for GDB7-22-TS and `-p` for Proparg-21-TS
- For example, for a train fraction of 0.8, with corresponding test and validation of 0.1 each, and 10-fold CV, the command `python src/run_all_fingerprints.py -c -g -p --CV=10 --train=0.8` will generate/load representations and run the models or load the results

## Language models
- A pre-trained BERT model is fine-tuned on the appropriate datasets (cyclo, gdb, or proparg)
- Here, the dataset is provided as an optional argument to `src/train_bert.py`
- To train on and then predict on the cyclo dataset for example, the command is `python src/train_bert.py -t -p --dataset='gdb' --test_size=0.2`
- The optional `--CV` flag determines whether to shuffle the train and predict datasets over the CV iterations
- The prediction MAE of 10 CV iterations are saved to `outs/cyclo_bert_pretrained/results.txt` (and the equivalent for the other datasets)

## Graph based models 
- The CGR model was run using the provided atom mapping, automatic atom mapping from rxnmapper, and random atom mapping 
- The data are in the csv files `data/gdb7-22-ts/ccsdtf12_dz.csv` for GDB7-22-TS,
  `data/cyclo/full_dataset.csv` for Cyclo-23-TS
  and `data/proparg/data.csv` for Proparg-21-TS
- SMILES with the provided ("true") atom mapping are
  in the `rxn_smiles` (Cyclo-23-TS, GDB7-22-TS) or `rxn_smiles_mapped` (Proparg-21-TS) columns of each csv file
- To run rxnmapper, run `src/mapper.py`
- Atom mapped SMILES from rxnmapper are in the `rxn_smiles_rxnmapper` column of each csv file
- Random atom maps are generated using `src/random_mapper.py` and 
  are in the `rxn_smiles_random` column of each csv file
- Python file specifying how each CGR model was run is `src/cgr.py`
- Results of the CGR runs can be found in `results` 

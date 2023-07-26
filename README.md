# Benchmarking of reaction representations

## Installation
- Due to the dependencies of the qml python package, numpy needs to be installed first and foremost, therefore a `requirements.txt` file is provided
- Recommend to first create a conda env: `conda create -n benchmark-rxn`
- `conda install pip`
- Then install requirements like `xargs -L 1 pip install < requirements.txt`

## 2D and 3D fingerprints
- 2D reps here are the DRFP and MFP
- 3D reps are SLATM, SPAHM$_b$, and $B^2R^2_l$, all in the difference variation for reactions
- To generate/load the reps and perform CV-fold cross-validated predictions with a train fraction of tr, the file `src/run_all_fingerprints.py` should be used
- Here, the datasets are specified as additional arguments, `-c` for cyclo, `-g` for gdb and `-p` for proparg
- For example, for a train fraction of 0.8, with corresponding test and validation of 0.1 each, and 10-fold CV, the command `python src/run_all_fingerprints.py -c -g -p --CV=10 --train=0.8` will generate/load representations and run the models or load the results

## Language models
- A pre-trained BERT model is fine-tuned on the appropriate datasets (cyclo, gdb, or proparg)
- Here, the dataset is provided as an optional argument to `src/train_bert.py`
- To train on and then predict on the cyclo dataset for example, the command is `python src/train_bert.py -t -p --dataset='gdb' --test_size=0.2`
- The optional `--CV` flag determines whether to shuffle the train and predict datasets over the CV iterations
- The prediction MAE of 10 CV iterations are saved to `outs/cyclo_bert_pretrained/results.txt` (and the equivalent for the other datasets)

## Graph based models 
- The CGR model was run using the provided atom mapping, automatic atom mapping from rxnmapper, and random atom mapping 
- To run rxnmapper, run `src/mapper.py` and then process the results using `src/process_maps.py`. The atom maps from mapper are already saved as `maps_dataset.pkl` in each dataset directory.
- csv files with atom mapped SMILES from rxnmapper are in `data/dataset/all_auto_mapped_rxns.csv`
- Random atom maps are generated using `src/random_mapper.py` and the corresponding csv in `data/dataset/random_mapped_rxns.csv`
- csv files were prepared for usage with `chemprop` amd saved as `submit_all_rxns.csv` or `submit_random_rxns.csv`
- Python files specifying how each CGR model was run are in `src`
- Results of the CGR runs can be found in `results` 
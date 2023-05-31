# Benchmarking of reaction representations

## Installation
- #TODO 
- Especially how to generate SPAHM 

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

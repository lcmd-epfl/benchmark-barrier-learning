import chemprop

arguments = ["--data_path",  "/home/vangerwe/atom-map/data/gdb7-22-ts/submit_all_rxns.csv",
	 "--dataset_type",  "regression",
	 "--target_columns", "barrier",
	 "--metric", "mae",
	 "--dropout", "0.05",
	 "--epochs", "300",
	 "--reaction",
	 "--num_folds",  "10",
	 "--save_dir", "/home/vangerwe/atom-map/results/gdb_all"]

args = chemprop.args.TrainArgs().parse_args(arguments)
print("Entering train for 10 folds...")
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
print("Mean score", mean_score, "std_score", std_score)

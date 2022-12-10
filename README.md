# Experimenting with Loss Functions in GNNs
## Patrick Flynn, Jack Oh, Yutong Wu

Exploring the Outlier Resistance of the Loss Function in a GNN

Our experiment is modifying the model from this paper: [Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf) (Fan, Wenqi, et al. "Graph Neural Networks for Social Recommendation." The World Wide Web Conference. ACM, 2019).

![architecture](assets/graphrec.png)

# Usage

1. Create a conda environment from the provided environment.yml file
```bash
conda env create -f environment.yml
```
   Activate the conda environment with
```bash
conda activate graphrec
```

2. Preprocess the dataset. Afer this, two pkl files named `dataset.pkl` and `list.pkl` should be generated in the folder of the dataset.
```bash
python preprocess.py --dataset Epinions
```

3. Create subsets of outliers using a specified z-score.
```bash
python subset_preprocess.py --threshold 0.5 1 1.5 2
```

4. Train models with different loss functions.
Already trained model checkpoints can be downloaded [here](https://drive.google.com/file/d/1uplhC3elHRqEZWyOZaZahj0gf9Dk8aZc/view?usp=share_link).
These files contain the best checkpoints for models trained with varying hyperparameters and lossfunctions.

If you want to train a new model:

For MSE:
```bash
python experiment.py --dataset_path=datasets/Epinions/ --loss_func=MSE
```

For Huber Loss:
```bash
python experiment.py --dataset_path=datasets/Epinions/ --loss_func=huber --delta={delta_value}
```


5. Testing 
If you want to run a test on all the models and all the test sets, running this command will find the folder **training_results** for models and "datasets/Epinions/" and run tests for all existing models and test sets.
```bash
python run_experiment.py
```

If you want to run single test:

For the entire test set:
```bash
python experiment.py --test --data_name dataset.pkl
```

For a subset test set:
```bash
python experiment.py --test_subset --data_name [test subset name]
```

6. To create a plot, you will need to run visualization.py. If you want to make a plot with different data, you will need to update "first_exp.csv" and "second_exp.csv".

To generate a plot for the first and second experiment:
```bash
python visualization.py
```

To generate a plot for the first experiment:
```bash
python visualization.py --plot 'exp1'
```

To generate plot for the second experiment:
```bash
python visualization.py --plot 'exp2'
```

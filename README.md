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

2. Preprocess the dataset. Two pkl files named dataset and list should be generated in the folder of the dataset.
```bash
python preprocess.py --dataset Epinions
```

3. Run main.py file to train the model. You can configure some training parameters through the command line. 
```bash
python main.py
```

4. Run main.py file to test the model.
```bash
python main.py --test
```

3. Create subsets of outliers using zscore.
```bash
python subset_preprocess --threshold 0.5 1 1.5 2
```

4. Train models with different loss functions.
Download the trained models in this link:
These files contains best checkpoints of trained models.

If you want to train new model:

For MSE:
```bash
python experiment.py --dataset_path=datasets/Epinions/ --loss_func=MSE
```

For Huber Loss:
```bash
python experiment.py --dataset_path=datasets/Epinions/ --loss_func=huber --delta={delta_value}
```


5. Run test 
If you want to run test on all the models and all the test sets, running this command will find the foldter "training_results" for models and "datasets/Epinions/" and run test for all exsting models and test sets.
```bash
python run_experiment.py
```

If you want to run single test:

For entire test set:
```bash
python experiment.py --test --data_name dataset.pkl
```

For subset of test sets:
```bash
python experiment.py --test_subset --data_name data_name
```

6. To create a plot, you will need to run visualization.py. If you want to make a plot with different data, you will need to update "first_exp.csv" and "second_exp.csv".

To generate plot for fist and second experiment.
```bash
python visualization.py
```

To generate plot for fist experiment.
```bash
python visualization.py --plot 'exp1'
```

To generate plot for second experiment.
```bash
python visualization.py --plot 'exp2'
```

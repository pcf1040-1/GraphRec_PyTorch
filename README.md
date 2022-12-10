# Experimenting with Loss Functions in GNNs
## Patrick Flynn, Jack Oh, Yutong Wu
An exploration of loss functions in GNNs using the following model: [Graph Neural Networks for Social Recommendation](https://arxiv.org/pdf/1902.07243.pdf) (Fan, Wenqi, et al. "Graph Neural Networks for Social Recommendation." The World Wide Web Conference. ACM, 2019).

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

import os

for data_name in os.listdir('datasets/Epinions/'):

    if data_name.startswith('dataset_subset'):
        print("Dataset: "+ data_name)
        os.system("python experiment.py --test_subset --data_name " + data_name)
    elif data_name=='dataset.pkl':
        print("Dataset: "+ data_name)
        os.system("python experiment.py --test --data_name dataset.pkl")
"""
@author: Jack Oh
"""
import os

file_list = os.listdir('datasets/Epinions/')
file_list.sort()
for data_name in file_list:

    if data_name.startswith('dataset_subset'):
        print("Dataset: "+ data_name)
        os.system("python experiment.py --test_subset --data_name " + data_name)
    elif data_name=='dataset.pkl':
        print("Dataset: "+ data_name)
        os.system("python experiment.py --test --data_name dataset.pkl")
    
    print('\n\n')
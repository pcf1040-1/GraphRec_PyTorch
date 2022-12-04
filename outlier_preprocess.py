# load the pickle file and get test set

# compute z score

# select the rows only if within the threshold

# save the new pickle file in path

import pandas as pd
import pickle
import numpy as np
import torch
import argparse
from scipy.stats import zscore
from dataloader import GRDataset
from torch.utils.data import DataLoader
from utils import collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/Epinions/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--threshold', default=1, nargs="+", type=float, help='threshold for zscore of subset, take a list as input')
parser.add_argument('--greater_or_less', default='less', help='determine to include the data if the absolute value of zscore is greater or less')
parser.add_argument('--data_name', default='dataset.pkl', help='name of the dataset pkl file to use')
args = parser.parse_args()
print(args)

def subset_outlier(path, fn, threshold,):
	with open(path + fn, 'rb') as f:
		print("open file")
		train_set = pickle.load(f)
		valid_set = pickle.load(f)
		test_set = pickle.load(f)

		df = pd.DataFrame(test_set, columns = ["user", "item", "rating"])

		# compute z score
		df_user = df.groupby(['user'])['rating'].mean().reset_index(name='average_rating')
		df_user['zscore'] = zscore(df_user['average_rating'])

		# include a user id only if within threshold
		subset = []
		for index, row in df_user.iterrows():
			if abs(row['zscore']) <= threshold:
				subset.append(row['user'])
		
		df = df.join(df_user.set_index('user'), on='user')

		if args.greater_or_less=='less':
			new_df = df[abs(df['zscore']) <= 1].reset_index()
		elif args.greater_or_less=='greater':
			new_df = df[abs(df['zscore']) <= 1].reset_index()

		new_df = new_df[['user', 'item', 'rating']]

		# save the pickle file
		new_fn = 'dataset_subset_' + args.greater_or_less + str(threshold) + '.pkl'
		with open(path + new_fn, 'wb') as n_f:
			print("Saving file in " + path + new_fn +'...')
			pickle.dump(new_df.values.tolist(), n_f, pickle.HIGHEST_PROTOCOL)


def main():
	# Run test for Epinions
	for thr in args.threshold:
		subset_outlier(path=args.dataset_path, fn=args.data_name, threshold=thr)
		
if __name__ == '__main__':
    main()

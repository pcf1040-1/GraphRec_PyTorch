# load the pickle file and get test set

# compute z score

# select the rows only if within the threshold

# save the new pickle file in path

import pandas as pd
import pickle
import numpy as np
import torch
from scipy.stats import zscore
from dataloader import GRDataset
from torch.utils.data import DataLoader
from utils import collate_fn
from model import GraphRec
from main import validate


def subset_outlier(path, fn, new_fn, threshold,):
	with open(path + fn, 'rb') as f:
		print("open file")
		train_set = pickle.load(f)
		valid_set = pickle.load(f)
		test_set = pickle.load(f)

		df = pd.DataFrame(test_set, columns = ["user", "item", "rating"])
		# print(df)

		# compute z score
		df_user = df.groupby(['user'])['rating'].mean().reset_index(name='average_rating')
		df_user['zscore'] = zscore(df_user['average_rating'])
		# print(df_user)

		# include a user id only if within threshold
		subset = []
		for index, row in df_user.iterrows():
			if abs(row['zscore']) <= threshold:
				subset.append(row['user'])
		
		# print(len(subset))

		df = df.join(df_user.set_index('user'), on='user')
		# print(df)

		new_df = df[abs(df['zscore']) <= 1].reset_index()
		# print(new_df)
		new_df = new_df[['user', 'item', 'rating']]
		# print(new_df)

		# save the pickle file
		with open(path + new_fn, 'wb') as n_f:
			pickle.dump(new_df, n_f, pickle.HIGHEST_PROTOCOL)


def test_subset(path, fn):
	print('Loading data...')
	with open(path + fn, 'rb') as f:
		test_set = pickle.load(f)
		
	with open(path + 'list.pkl', 'rb') as f:
		u_items_list = pickle.load(f)
		u_users_list = pickle.load(f)
		u_users_items_list = pickle.load(f)
		i_users_list = pickle.load(f)
		(user_count, item_count, rate_count) = pickle.load(f)
	
	test_set = test_set.values.tolist()	# convert dataframe to list (to input correct format for GRdataset)

	test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
	test_loader = DataLoader(test_data, batch_size = 256, shuffle = False, collate_fn = collate_fn)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = GraphRec(user_count+1, item_count+1, rate_count+1, 64).to(device)
	
	# testing with subset
	print('Load checkpoint and testing...')
	ckpt = torch.load('latest_checkpoint.pth.tar')
	model.load_state_dict(ckpt['state_dict'])
	mae, rmse = validate(test_loader, model)
	print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))


def main():
	# Run test for Epinions
	subset_outlier(path='datasets/Epinions/', fn='dataset.pkl', new_fn='dataset_subset_1.pkl', threshold=1)
	test_subset(path='datasets/Epinions/', fn='dataset_subset_1.pkl')

	# Run test for Ciao
	# subset_outlier(path='datasets/Ciao/', fn='dataset.pkl', new_fn='dataset_subset_1.pkl', threshold=1)
	# test_subset(path='datasets/Ciao/', fn='dataset_subset_1.pkl')
	
if __name__ == '__main__':
    main()
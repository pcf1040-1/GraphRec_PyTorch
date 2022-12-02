# load the pickle file and get test set

# compute z score

# add the line only if zscore is in threshold

# save the new pickle file in path

import pandas as pd

import pickle
import numpy as np
from scipy.stats import zscore

from dataloader import GRDataset
from torch.utils.data import DataLoader
from utils import collate_fn

import argparse
import torch
from model import GraphRec
from main import validate


def subset_outlier(path, fn, threshold):
	with open(path + fn, 'rb') as f:
		print("open file")
		train_set = pickle.load(f)
		valid_set = pickle.load(f)
		test_set = pickle.load(f)

		df = pd.DataFrame(test_set, columns = ["user", "item", "rating"])
		print(df)

		# compute z score
		df_user = df.groupby(['user'])['rating'].mean().reset_index(name='average_rating')
		df_user['zscore'] = zscore(df_user['average_rating'])
		print(df_user)

		# include a user id only if within threshold
		subset = []
		for index, row in df_user.iterrows():
			if abs(row['zscore']) <= threshold:
				subset.append(row['user'])
		
		print(len(subset))

		df = df.join(df_user.set_index('user'), on='user')
		print(df)

		new_df = df[abs(df['zscore']) <= 1]
		print(new_df)
		new_df = new_df[['user', 'item', 'rating']]
		print(new_df)


		with open(path + '/dataset_zscore_' + str(threshold) + '.pkl', 'wb') as n_f:
			pickle.dump(new_df, n_f, pickle.HIGHEST_PROTOCOL)



# some arg parser stuff (copied from main)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/Epinions/', help='dataset directory path: datasets/Ciao/Epinions')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=30, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()

def main():
	subset_outlier('datasets/Epinions/', 'dataset.pkl', 1)

	print('Loading data...')
	with open('datasets/Epinions/' + 'dataset.pkl', 'rb') as f:
		test_set = pickle.load(f)
		
	with open('datasets/Epinions/' + 'list.pkl', 'rb') as f:
		u_items_list = pickle.load(f)
		u_users_list = pickle.load(f)
		u_users_items_list = pickle.load(f)
		i_users_list = pickle.load(f)
		(user_count, item_count, rate_count) = pickle.load(f)
	
	test_data = GRDataset(test_set, u_items_list, u_users_list, u_users_items_list, i_users_list)
	test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = GraphRec(user_count+1, item_count+1, rate_count+1, 64).to(device)
	
	# testing with subset
	print('Load checkpoint and testing...')
	ckpt = torch.load('latest_checkpoint.pth.tar')
	model.load_state_dict(ckpt['state_dict'])
	mae, rmse = validate(test_loader, model)
	print("Test: MAE: {:.4f}, RMSE: {:.4f}".format(mae, rmse))

if __name__ == '__main__':
    main()
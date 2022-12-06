"""
@author: Yutong Wu
This file runs all the visualization for our expriments,
including z score distribution, and MAE loss plots
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot', default='all', help='plots to generate: all/zscore/exp1/exp2')
args = parser.parse_args()
print(args)

# Reads the Epinion dataset and plot the z score distribution for a user's rating
# A user's rating is calculated as the average of all the ratings of that user
def plt_z_score():
    df = pd.read_csv('./datasets/Epinions/ratings_data.txt', 
    sep=' ', header=None, names=["user", "item", "rating"])
    df_user = df.groupby(['user'])['rating'].mean().reset_index(name='rating')
    df_user['zscore'] = zscore(df_user['rating'])
    plt.hist(df_user['zscore'],edgecolor='black')
    plt.xlabel("z score for users")
    plt.ylabel("Frequency")
    plt.title("Z score Distribution")
    plt.savefig("exp_result/z_score_dist.pdf")
    plt.close()

# Reads the first experiment loss file, outputs 2 plots in the /exp_result folder
# One plot for dataset with > z scores, and the other for < z scores
def plt_first_exp():
    df = pd.read_csv('exp_result/first_exp.csv', sep = ',')
    color = ['#4a708b', '#e9b900', '#556a0c', '#be3455', '#7f5a83']
    count = 1
    # First plot: greater than, MAE
    for idx in range(4, 8):
        if idx == 4:
            plt.plot(df['Delta'], df['MAE for Complete Set'], '-o', 
            color = color[0], label = "Entire Set")
        name = df.columns[idx]
        col = df.iloc[:, idx]
        plt.plot(df['Delta'], col, '-o',color = color[count], label = name.split(" ")[2])
        count += 1
    plt.xlabel("Delta")
    plt.ylabel("MAE")
    plt.title("MAE on Entire Set and Subsets Greater Than Z Scores")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("exp_result/first_exp_MAE_Greater.pdf")
    plt.close()

    # Second plot: less than, MAE
    count = 1
    for idx in range(8, 12):
        if idx == 8:
            plt.plot(df['Delta'], df['MAE for Complete Set'], '-o', 
            color = color[0], label = "Entire Set")
        name = df.columns[idx]
        col = df.iloc[:, idx]
        plt.plot(df['Delta'], col, '-o',color = color[count], label = name.split(" ")[2])
        count += 1
    plt.xlabel("Delta")
    plt.ylabel("MAE")
    plt.title("MAE on Entire Set and Subsets Less Than Z Scores")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("exp_result/first_exp_MAE_Less.pdf")
    plt.close()

# Plot the second experiment, this is the plot with the loss on the entire dataset only
def plt_second_exp():
    df = pd.read_csv('exp_result/second_exp.csv', sep=',')
    plt.plot(df['Delta'], df['MAE for Complete Set'], '-o', 
    color = '#4a708b', label = "Entire Set")       
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Delta")
    plt.ylabel("MAE")
    plt.title("Entire Set Loss for Second Experiment")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("exp_result/second_exp.pdf")
    plt.close()

def main():
    if args.plot=='zscore':
        plt_z_score()
    elif args.plot=='exp1':
        plt_first_exp()
    elif args.plot=='exp2':
        plt_second_exp()
    elif args.plot=='all':
        plt_z_score()
        plt_first_exp()
        plt_second_exp()
    

if __name__ == '__main__':
    main()

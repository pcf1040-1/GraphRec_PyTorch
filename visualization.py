import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import numpy as np
from scipy.stats import zscore

def plot_z_score(df):
    df_user = df.groupby(['user'])['rating'].mean().reset_index(name='rating')
    df_user['zscore'] = zscore(df_user['rating'])
    plt.hist(df_user['zscore'],edgecolor='black')
    plt.xlabel("z score for users")
    plt.ylabel("Frequency")
    plt.title("Z score Distribution")
    plt.savefig("z_score_dist.pdf")
    plt.close()

def read_epinion_data():
    df = pd.read_csv('./datasets/Epinions/ratings_data.txt', 
    sep=' ', header=None, names=["user", "item", "rating"])
    return df

def plt_loss_dist():
    df = pd.read_csv('loss_data.csv', sep = ',')
    color = ['#4a708b', '#e9b900', '#556a0c', '#be3455', '#7f5a83']
    count = 1
    # greater than, MAE
    for idx in range(5, 13, 2):
        if idx == 5:
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
    plt.savefig("MAE_Greater.pdf")
    plt.close()
    count = 1
    # greater than, RMSE
    for idx in range(6, 14, 2):
        if idx == 6:
            plt.plot(df['Delta'], df['RMSE for Complete Set'], '-o', 
            color = color[0], label = "Entire Set")
        name = df.columns[idx]
        col = df.iloc[:, idx]
        plt.plot(df['Delta'], col, '-o',color = color[count], label = name.split(" ")[2])
        count += 1
    plt.xlabel("Delta")
    plt.ylabel("RMSE")
    plt.title("RMSE on Entire Set and Subsets Greater Than Z Scores")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("RMSE_Greater.pdf")
    plt.close()
    # less than, MAE
    count = 1
    for idx in range(13, 21, 2):
        if idx == 13:
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
    plt.savefig("MAE_Less.pdf")
    plt.close()
    # less than, RMSE
    count = 1
    for idx in range(14, 21, 2):
        if idx == 14:
            plt.plot(df['Delta'], df['RMSE for Complete Set'], '-o', 
            color = color[0], label = "Entire Set")
        name = df.columns[idx]
        col = df.iloc[:, idx]
        plt.plot(df['Delta'], col, '-o',color = color[count], label = name.split(" ")[2])
        count += 1
    plt.xlabel("Delta")
    plt.ylabel("RMSE")
    plt.title("RMSE on Entire Set and Subsets Less Than Z Scores")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig("RMSE_Less.pdf")
    plt.close()

def main():
    df = read_epinion_data()
    plot_z_score(df)
    plt_loss_dist()
    

if __name__ == '__main__':
    main()

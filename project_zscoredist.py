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

def read_data(Ciao = False, Epinion = False):
    if Epinion:
        df = pd.read_csv('/home/stu11/s8/yw6228/Desktop/ml2022/GraphRec_PyTorch/datasets/Epinions/ratings_data.txt', 
        sep=' ', header=None, names=["user", "item", "rating"])
        return df
    elif Ciao:
        mat = loadmat('/home/stu11/s8/yw6228/Desktop/ml2022/GraphRec_PyTorch/datasets/Ciao/rating.mat')
        mat = {k:v for k, v in mat.items() if k[0] != '_'}
        df = pd.DataFrame({k: np.array(v).flatten() for k, v in mat.items()})
        return df

def main():
    df = read_data(Epinion = True)
    plot_z_score(df)

if __name__ == '__main__':
    main()
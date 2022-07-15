import numpy as np 
import pandas as pd
import argparse

def list_of_mask_df(saving_path, n_random_mask, seed=None):
    np.random.seed(seed)

    #seed_list = np.random.choice(100000, n_random_mask,replace=False,).tolist()
    seed_list = np.random.choice(100000, n_random_mask+3,replace=False,).tolist()

    #mask_seed_list = [None]*3 + seed_list
    mask_seed_list = seed_list
    mask_type_list = ["caipiranha","2D_uniform",'uniform'] +['random']*n_random_mask
    
    df = pd.DataFrame({"mask_typ":mask_type_list, "mask_seed": mask_seed_list})
    
    df.to_csv(saving_path, sep=',', index=False)
    
#list_of_mask_df(saving_path="C:/Users/jeane/argpase comprehension/test_parallel_4/list_of_mask_df.csv", n_random_mask=2, seed=123)

parser = argparse.ArgumentParser()
parser.add_argument("saving_path",
    metavar='csv_file',
    default='None',
    help='csv_file where are listed the type of mask to generate and their random seed ')

parser.add_argument("n_random_mask", type=int, help="number of random masks")
parser.add_argument("seed", type=int, help="number of random masks", default=None)
args = parser.parse_args()

if __name__ == '__main__':
    list_of_mask_df(args.saving_path, args.n_random_mask, args.seed)

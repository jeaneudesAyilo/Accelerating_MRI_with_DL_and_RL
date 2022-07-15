import glob
import os
import numpy as np
import pandas as pd
import re

##here, we want to gather the different results

grid_search_result_path = "./result"

os.chdir(grid_search_result_path)

csvfiles = glob.glob("./*/result_grid_search.csv")
csvfiles.sort(key=lambda x: os.path.getmtime(x))

general_grid_search_result_df = []


for file in csvfiles:
    
    try:
        df = pd.read_csv(file,"\t")
    except:
        pass
    
    general_grid_search_result_df.append(df)
    

general_grid_search_result_df = pd.concat(general_grid_search_result_df, ignore_index=True)

general_grid_search_result_df.to_csv("./general_grid_search_result_df.csv", sep='\t')   

##select the best result according to the loss of the test (mse)

idx = general_grid_search_result_df.groupby(['bloc'])['test_loss'].transform(min) == general_grid_search_result_df['test_loss']

best_result_df =general_grid_search_result_df[idx]

best_result_df.to_csv("./best_result_df.csv", sep='\t')   

print("***************FIN***************")

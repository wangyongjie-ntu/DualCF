#Filename:	synthetic.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 26 Jul 2021 02:56:26 

import torch
import dice_ml
import pandas as pd
import numpy as np

if __name__ == "__main__":

    dquery = pd.read_csv("synthetic_query.csv")
    dtrain = pd.read_csv("synthetic_train.csv")
    columns = dtrain.columns.tolist()
    d = dice_ml.Data(dataframe = dtrain, continuous_features = columns[0:2], outcome_name = 'Y')

    backend = 'PYT'
    model_path = "synthetic.pt"
    m = dice_ml.Model(model_path = model_path, backend = backend)
    exp = dice_ml.Dice(d, m)

    cf_df = None

    index = []

    for i in range(len(dquery)):
        print(i)
        query_instance = dquery.iloc[i, 0:2].to_dict()
        dice_exp = exp.generate_counterfactuals(query_instance, total_CFs= 1, desired_class='opposite', verbose = False, posthoc_sparsity_algorithm=None,
                                               max_iter= 5000, learning_rate=0.05, init_near_query_instance= False, loss_converge_maxiter = 2)

        if not dice_exp.cf_examples_list[0].final_cfs_df.empty:
            index.append(i)
            if cf_df is None:
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            else:
                cf_df = pd.concat((cf_df, dice_exp.cf_examples_list[0].final_cfs_df))

    cf_df.to_csv("synthetic-dice.csv")
    np.save("index.npy", index)

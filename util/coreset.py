#Filename:	coreset.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Kam 09 Des 2021 02:36:51 

import random
import numpy as np
from scipy.spatial.distance import pdist, squareform


def find_core_set(query_x, K):

    pairwise_dists = squareform(pdist(query_x, 'euclidean'))
    # random select one
    seed = random.choice(range(len(query_x)))
    idx_list = []
    idx_list.append(seed)
    
    # select the second one
    # idx = np.argmax(pairwise_dists[seed])
    # idx_list.append(idx)

    for i in range(K-1):
        subrows = pairwise_dists[idx_list]
        min_col = subrows.min(0)
        idx = np.argmax(min_col)
        idx_list.append(idx)

    return idx_list

if __name__  == "__main__":

    test = np.random.randint(0, 5, (5, 4))
    print(test)
    a = find_core_set(test, 2)
    print(test[a])


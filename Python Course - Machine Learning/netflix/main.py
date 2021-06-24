import numpy as np
import kmeans
import common
import naive_em
import em
import pandas as pd

X = np.loadtxt("netflix_incomplete.txt")
seed = 0

# TODO: Your code here
K = [1,12]
seeds = [0,1,2,3,4]
costs = pd.DataFrame(index = K, columns=seeds)

for k in K:
    for seed in seeds:
        mixture, post = common.init(X,k,seed)
        mixture, post, cost = em.run(X, mixture, post)
        #common.plot(X, mixture, post, cost)
        costs.at[k,seed] = cost

print(costs.max(axis=1))

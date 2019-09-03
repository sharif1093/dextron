import numpy as np

#########################################
## Distributing commands based on load ##
#########################################
def partition_greedy(weights, n_partitions):
    def sum(partition, weights):
        s = 0
        for e in partition:
            s += weights[e]
        return s
    def argmin(li):
        index = list(range(len(li)))
        return min(index, key=li.__getitem__)

    index = list(range(len(weights)))
    items = list(zip(index, weights))
    
    plist = [[] for i in range(n_partitions)]
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
    
    
    for k in items_sorted:
        sums = []
        for i in range(n_partitions):
            sums += [sum(plist[i], weights)]
        index_min = argmin(sums)
        plist[index_min] += [k[0]]
    sums = []
    for i in range(n_partitions):
        sums += [sum(plist[i], weights)]
    score = np.std(sums) / sum(index, weights)
    return plist, sums, score

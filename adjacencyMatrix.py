import csv
import numpy as np

#class readCSV():
    
def create_adjacency_mtx(graph):
    mtx = np.empty((6000,6000))
    mtx.fill(0)

    for row in graph:
        arr = row[0].split(",")
        mtx[int(arr[0])-1][int(arr[1])-1] = 1
        mtx[int(arr[1])-1][int(arr[0])-1] = 1
    
    return mtx
import functions
import adjacencyMatrix as adjMtx
import numpy as np
import scipy.spatial.distance as dist
#import Seed.csv as Seed
#import Graph.csv as Graph
#import Extracted_features.csv as EF
import csv
from sklearn.cluster import SpectralClustering
#import matplotlib.pyplot as plt
#import sklearn.manifold as spectralEmb
#import networkx as nx
#import scipy.sparse.linalg as linalg
import logging
from sklearn.cluster import KMeans
class main():
    
    with open('/Users/kat/Desktop/Kaggle/Graph.csv', 'rb') as csvfile1:
        graphreader = csv.reader(csvfile1, delimiter=' ', quotechar='|')            
        adjgraph = np.empty((6000,6000))
        adjgraph.fill(0)
        for row in graphreader:
            arr = row[0].split(",")
            adjgraph[int(arr[0])-1][int(arr[1])-1] = 1
            adjgraph[int(arr[1])-1][int(arr[0])-1] = 1
  
    # get features into matrix
    with open('/Users/kat/Desktop/reduced_1035_all_points.csv', 'rb') as csvfile3:
        EF = csv.reader(csvfile3, delimiter=' ', quotechar='|')
        newEF = []
        for row in EF:
            arr = row[0].split(",")
            arr2 = np.asarray(arr)
            arr3 = arr2.astype(np.float)
            newEF.append(arr3)
        
    
    with open('/Users/kat/Desktop/reduced_3000.csv', 'rb') as csvfile3:
        EF = csv.reader(csvfile3, delimiter=' ', quotechar='|')
        adj_new = []
        for row in EF:
            arr = row[0].split(",")
            arr2 = np.asarray(arr)
            arr3 = arr2.astype(np.float)
            adj_new.append(arr3)
  
     
    # spectral clustering on adjacency matrix       
    spectral = SpectralClustering(10, affinity="precomputed")
    new_plot = spectral.fit_predict(adj_new) #6000 x 1 Array with cluster labels
    
    matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    clusters = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # get cluster matchings for first 60 points
    with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
        seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
        for row in seedreader:
            arr = row[0].split(",")
            findClust = new_plot[int(arr[0])-1]
            matching[int(arr[1])].append([int(arr[0]),newEF[int(arr[0])-1], findClust])
    
    for i in range(1,6001):
        findClust = new_plot[i-1]
        clusters[findClust].append(newEF[i-1])
        
    for i in range(10):
        print "item is " + str(i)
        for item in matching[i]:
            print item[2]
    
                                                                                                                  
    filtered_features = []
    filtered_features_idx = []
    cluster_5_digit_6 = []
    cluster_8_digit_1 = []
    
    
    for i in range(len(new_plot)):
        if new_plot[i] == 5:
            cluster_5_digit_6.append([i+1, red_pca[i]])
        elif new_plot[i] == 8:
            cluster_8_digit_1.append([i+1,red_pca[i]])
        else:
            filtered_features.append(red_pca[i])
            filtered_features_idx.append(i+1)
    
    cluster_centers_pca_8_clusters = []
    for i in range(10):
         newarray = []
         if i == 1 or i == 6:
             pass
         else: 
            for j in range(len(matching[i])):
                newarray.append(np.asarray(matching[i][j][1]))         
            newa = np.asarray(newarray)
            cluster_centers_pca_8_clusters.append(newa.mean(axis=0))
    
    cluster_centers_pca_8_clusters = np.asarray(cluster_centers_pca_8_clusters)
            
    kmeans_8 = KMeans(n_clusters=8, init=cluster_centers_pca_8_clusters).fit_predict(filtered_features)
    
    updated_matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
        seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
        for row in seedreader:
            arr = row[0].split(",")
            if int(arr[1]) == 1 or int(arr[1]) == 6:
                pass
            else:
                try:
                    idx = filtered_features_idx.index(int(arr[0]))
                    updated_matching[int(arr[1])].append([int(arr[0]), red_pca[int(arr[0])-1], kmeans_8[idx]])
                except ValueError:
                    pass
                 
    finalmatches = {0:4, 1:2, 2:5, 3:9, 4:7, 5:3, 6:0, 7:8, 8:1, 9:6}
    
    adjustedcluster = {}
    for i in range(10):
        index = finalmatches[i]
        adjustedcluster[i] = clusters[index]
    
    cluster_centers = []  
    
    for i in range(10):
        newa = np.asarray(adjustedcluster[i])
        cluster_centers.append(newa.mean(axis=0))
    
    
    #for i in range(10):
    #     newarray = []
    #     if i == 1 or i == 6:
    #         pass
    #     else:
    #        for j in range(len(updated_matching[i])):
    #            newarray.append(np.asarray(updated_matching[i][j][1]))         
    #        newa = np.asarray(newarray)
    #        cluster_centers_kmeans.append(newa.mean(axis=0))
    
    digit_6_feat_pca = []
    for item in cluster_5_digit_6:
        digit_6_feat_pca.append(item[1])

    digit_1_feat_pca = []
    for item in cluster_8_digit_1:
        digit_1_feat_pca.append(item[1])
    
    digit_6_feat_pca = np.asarray(digit_6_feat_pca)
    digit_6_centroid = digit_6_feat_pca.mean(axis=0)
    digit_1_feat_pca = np.asarray(digit_1_feat_pca)
    digit_1_centroid = digit_1_feat_pca.mean(axis=0)

    cluster_centers_kmeans.insert(1, digit_1_centroid)
    cluster_centers_kmeans.insert(6, digit_6_centroid)     
                   
    finalclusters = [[0 for i in range(2)] for j in range(4001)]
    finalclusters[0][0] = 'Id'
    finalclusters[0][1] = 'Label'
    
    for i in range(1,4001):
         finalclusters[i][0] = 6000 + i
         newdist = []
         for j in range(10):
             newdist.append(dist.euclidean(newEF[i+5999], cluster_centers[j]))
         label = np.argmin(newdist)
         finalclusters[i][1] = label
                
                        
    with open('submission7.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(finalclusters)
    
          
    #g = nx.from_numpy_matrix(adjgraph)
    #plt.plot(g)
    #plt.show()
    #spect = spectralEmb.SpectralEmbedding(10, affinity="precomputed")
    #new_plot = spect.fit_transform(adjgraph)
   
    #kmeans = clust.Kmeans(n_clusters=10).fit(new_plot)
    
    #labels = kmeans.labels_ 
    #centers = kmeans.cluster_centers_
    
    #print labels
    #plt.plot(new_plot)
    #plt.show()
    
    #plt.scatter(new_plot)
    #plt.show()
    
    
    #newEF=PCA(EF);



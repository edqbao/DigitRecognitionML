from sklearn.decomposition import PCA, KernelPCA
import csv
import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class PCA_and_GMM():
    
    ## Create adjacency matrix
    #with open('/Users/kat/Desktop/Kaggle/Graph.csv', 'rb') as csvfile1:
    #    graphreader = csv.reader(csvfile1, delimiter=' ', quotechar='|')            
    #    adjgraph = np.empty((6000,6000))
    #    adjgraph.fill(0)
    #    for row in graphreader:
    #        arr = row[0].split(",")
    #        adjgraph[int(arr[0])-1][int(arr[1])-1] = 1
    #        adjgraph[int(arr[1])-1][int(arr[0])-1] = 1
    
    # Get features data into newEF matrix
    with open('/Users/kat/Desktop/Kaggle/Extracted_features.csv', 'rb') as csvfile3:
        EF = csv.reader(csvfile3, delimiter=' ', quotechar='|')
        newEF = []
        for row in EF:
            arr = row[0].split(",")
            arr2 = np.asarray(arr)
            arr3 = arr2.astype(np.float)
            newEF.append(arr3)
    
    # PCA reduce features data to 800 dim (instead of 1084)
    pca = PCA(n_components=900)
    red_pca = pca.fit_transform(newEF)
    
  
    GMM = GaussianMixture(n_components = 10)
    GMM = GMM.fit(red_pca)
    GMM_labels = GMM.predict(red_pca)
    
    matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # get cluster matchings for first 60 points
    with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
        seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
        for row in seedreader:
            arr = row[0].split(",")
            findClust = GMM_labels[int(arr[0])-1]
            matching[int(arr[1])].append([int(arr[0]),red_pca[int(arr[0])-1], findClust])
    
    clusters = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # Get points of each cluster
    for i in range(1,10001):
        findClust = GMM_labels[i-1]
        clusters[findClust].append(red_pca[i-1])
     
    for i in range(10):
        print "item is " + str(i)
        for item in matching[i]:
            print item[2]
    
    finalmatches = {0:4, 1:7, 2:9, 3:2, 4:0, 5:8, 6:6, 7:3, 8:1, 9:5}
    
    # match clusters to digits
    adjustedcluster = {}
    for i in range(10):
        index = finalmatches[i]
        adjustedcluster[i] = clusters[index]                  
  
    cluster_centers = []   
    
    # calculate the cluster center for each cluster (digit)
    for i in range(10):
        newa = np.asarray(adjustedcluster[i])
        cluster_centers.append(newa.mean(axis=0))
    
    finalclusters = [[0 for i in range(2)] for j in range(4001)]
    finalclusters[0][0] = 'Id'
    finalclusters[0][1] = 'Label'
    
    for i in range(1,4001):
         finalclusters[i][0] = 6000 + i
         newdist = []
         for j in range(10):
             newdist.append(dist.euclidean(red_pca[i+5999], cluster_centers[j]))
         label = np.argmin(newdist)
         finalclusters[i][1] = label
                                    
    with open('submission11.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(finalclusters) 
    
    
    
    
    #filtered_features = []
    #filtered_features_idx = []
    #
    ## save clusters for digits 1 and 6
    #cluster_2_digit_1 = adjustedcluster[1]
    #cluster_7_digit_6 = adjustedcluster[6]
    #
    ## filter out clusters for digits 1 and 6
    #for i in range(len(new_plot)):
    #    if not new_plot[i] == 2 and not new_plot[i] == 7:
    #        filtered_features.append(red_pca[i])
    #        filtered_features_idx.append(i+1)
    #
    #centroids_pca_8_clusters = []
    #
    ### get initial centroids of the 8 digits based on seed
    ##for i in range(10):
    ##     newarray = []
    ##     if not i == 1 and not i == 6:
    ##        for j in range(len(matching[i])):
    ##            newarray.append(np.asarray(matching[i][j][1]))         
    ##        newa = np.asarray(newarray)
    ##        centroids_pca_8_clusters.append(newa.mean(axis=0))
    ##
    ##centroids_pca_8_clusters = np.asarray(centroids_pca_8_clusters)
    #
    #GMM = GaussianMixture(n_components = 8)
    #GMM = GMM.fit(filtered_features)
    #GMM_labels = GMM.predict(filtered_features)
    #
    #GMM_matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    #
    #with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
    #    seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
    #    for row in seedreader:
    #        arr = row[0].split(",")
    #        if not int(arr[1]) == 1 and not int(arr[1]) == 6:
    #            try:
    #                idx = filtered_features_idx.index(int(arr[0]))
    #                GMM_matching[int(arr[1])].append([int(arr[0]), red_pca[int(arr[0])-1], GMM_labels[idx]])
    #            except ValueError:
    #                pass
    #
    #for i in range(10):
    #    print "item is " + str(i)
    #    for j in range(len(GMM_matching[i])):
    #        print GMM_matching[i][j][2]
    #         
    #clusters_kmeans = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    ## Get points of each cluster from kmeans
    #for i in range(len(GMM_labels)):
    #    findClust = GMM_labels[i]
    #    idx = filtered_features_idx[i]
    #    clusters_kmeans[findClust].append(red_pca[idx-1])
       
#    finalmatches_kmeans = {0:2, 2:3, 3:1, 4:7, 5:0, 7:6, 8:6, 9:7}  
#    
#    # match clusters to digits
#    adjustedcluster_kmeans = {}
#    for i in range(10):
#        if not i == 1 and not i == 6:
#            index = finalmatches_kmeans[i]
#            adjustedcluster_kmeans[i] = clusters_kmeans[index] 
#    
#    # get features of digit 1 from spectral
#    cluster_2_digit_1 = np.asarray(cluster_2_digit_1)
#    digit_1_centroid = cluster_2_digit_1.mean(axis=0)
#
#    # get features of digit 6 from spectral        
#    cluster_7_digit_6 = np.asarray(cluster_7_digit_6)
#    digit_6_centroid = cluster_7_digit_6.mean(axis=0)
#     
#    cluster_centers = []  
#    
# 
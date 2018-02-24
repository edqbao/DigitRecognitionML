from sklearn.decomposition import PCA
import csv
import numpy as np
import scipy.spatial.distance as dist
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import scipy

class PCA_and_Spectral():
    
    # Create adjacency matrix
    with open('/Users/kat/Desktop/Kaggle/Graph.csv', 'rb') as csvfile1:
        graphreader = csv.reader(csvfile1, delimiter=' ', quotechar='|')            
        adjgraph = np.empty((6000,6000))
        adjgraph.fill(0)
        for row in graphreader:
            arr = row[0].split(",")
            adjgraph[int(arr[0])-1][int(arr[1])-1] = 1
            adjgraph[int(arr[1])-1][int(arr[0])-1] = 1
    
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
    pca = PCA(n_components=800)
    red_pca = pca.fit_transform(newEF)
    
    # spectral clustering on adjacency matrix       
    spectral = SpectralClustering(10, affinity="precomputed")
    new_plot = spectral.fit_predict(adjgraph) #6000 x 1 Array with cluster labels
    
    matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # get cluster matchings for first 60 points
    with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
        seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
        for row in seedreader:
            arr = row[0].split(",")
            findClust = new_plot[int(arr[0])-1]
            matching[int(arr[1])].append([int(arr[0]),red_pca[int(arr[0])-1], findClust])
    
    clusters = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # Get points of each cluster
    for i in range(1,6001):
        findClust = new_plot[i-1]
        clusters[findClust].append(red_pca[i-1])
     
    for i in range(10):
        print "item is " + str(i)
        for item in matching[i]:
            print item[2]
    
    finalmatches = {0:8, 1:2, 2:9, 3:4, 4:0, 5:6, 6:3, 7:5, 8:1, 9:7}
    
    # match clusters to digits
    adjustedcluster = {}
    for i in range(10):
        index = finalmatches[i]
        adjustedcluster[i] = clusters[index]                  
    
    filtered_features = []
    filtered_features_idx = []
    
    # save clusters for digits 1 and 6
    cluster_2_digit_1 = adjustedcluster[1]
    cluster_3_digit_6 = adjustedcluster[6]
    
    # filter out clusters for digits 1 and 6
    for i in range(len(new_plot)):
        if not new_plot[i] == 2 and not new_plot[i] == 3:
            filtered_features.append(red_pca[i])
            filtered_features_idx.append(i+1)
    
    centroids_pca_8_clusters = []
    
    # get initial centroids of the 8 digits based on seed
    for i in range(10):
         newarray = []
         if not i == 1 and not i == 6:
            for j in range(len(matching[i])):
                newarray.append(np.asarray(matching[i][j][1]))         
            newa = np.asarray(newarray)
            centroids_pca_8_clusters.append(newa.mean(axis=0))
    
    centroids_pca_8_clusters = np.asarray(centroids_pca_8_clusters)
        
    # do kmeans to clean up 8 clusters
    kmeans_8 = KMeans(n_clusters=8, init=centroids_pca_8_clusters).fit_predict(filtered_features)
    
    kmeans_matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    with open('/Users/kat/Desktop/Kaggle/Seed.csv', 'rb') as csvfile2:
        seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
        for row in seedreader:
            arr = row[0].split(",")
            if not int(arr[1]) == 1 and not int(arr[1]) == 6:
                try:
                    idx = filtered_features_idx.index(int(arr[0]))
                    kmeans_matching[int(arr[1])].append([int(arr[0]), red_pca[int(arr[0])-1], kmeans_8[idx]])
                except ValueError:
                    pass
    
    for i in range(10):
        print "item is " + str(i)
        for j in range(len(kmeans_matching[i])):
            print kmeans_matching[i][j][2]
             
    clusters_kmeans = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    # Get points of each cluster from kmeans
    for i in range(len(kmeans_8)):
        findClust = kmeans_8[i]
        idx = filtered_features_idx[i]
        clusters_kmeans[findClust].append(red_pca[idx-1])
       
    finalmatches_kmeans = {0:0, 2:1, 3:2, 4:3, 5:4, 7:5, 8:6, 9:7}  
    
    # match clusters to digits
    adjustedcluster_kmeans = {}
    for i in range(10):
        if not i == 1 and not i == 6:
            index = finalmatches_kmeans[i]
            adjustedcluster_kmeans[i] = clusters_kmeans[index] 
    
    adjustedcluster_kmeans[1] = cluster_2_digit_1
    adjustedcluster_kmeans[6] = cluster_3_digit_6
    
    # get features of digit 1 from spectral
    cluster_2_digit_1 = np.asarray(cluster_2_digit_1)
    digit_1_centroid = cluster_2_digit_1.mean(axis=0)

    # get features of digit 6 from spectral        
    cluster_3_digit_6 = np.asarray(cluster_3_digit_6)
    digit_6_centroid = cluster_3_digit_6.mean(axis=0)
     
    cluster_centers = []  
    
    # calculate the cluster center for each cluster (digit)
    for i in range(10):
        if i == 1:
            cluster_centers.append(digit_1_centroid)
        elif i == 6:
            cluster_centers.append(digit_6_centroid)
        else:
            newa = np.asarray(adjustedcluster_kmeans[i])
            cluster_centers.append(newa.mean(axis=0))
    
    finalclusters = [[0 for i in range(2)] for j in range(4001)]
    finalclusters[0][0] = 'Id'
    finalclusters[0][1] = 'Label'
    
    
    withinclusterdistance = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
    #for each cluster
    for i in range(10): 
        for k in range(len(adjustedcluster_kmeans[i])):
	   withinclusterdistance[i].append(dist.euclidean(adjustedcluster_kmeans[i][k], cluster_centers[i])) 

    for i in range(1,4001):
	finalclusters[i][0] = 6000 + i
	newdist = []
	label=100
	
	for j in range(10):
	   distancetocentroid = dist.euclidean(red_pca[i+5999], cluster_centers[j])
	   newdist.append(distancetocentroid)
	   meanX = np.mean(withinclusterdistance[j])
	   varX = np.var(withinclusterdistance[j])
	   n = len(withinclusterdistance[j])
	   LB = meanX - scipy.stats.norm.ppf(0.93) * np.sqrt(varX/n) #lowerbound #93% confidence lower bound
	   UB = meanX + scipy.stats.norm.ppf(0.93) * np.sqrt(varX/n) #upperbound #93% confidence upper bound
	   if (distancetocentroid < UB and distancetocentroid > LB): #check if point lies in CI of cluster J
	       label = j
			####pass? exit loop #####
	if (label == 100): #if the point does not lie within the confidence interval, take the min distance
            label = np.argmin(newdist)
	finalclusters[i][1] = label
                                    
    with open('submission14_conf_interval.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(finalclusters) 
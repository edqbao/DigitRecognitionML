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
    
    finalmatches = {0:9, 1:1, 2:8, 3:6, 4:5, 5:2, 6:3, 7:4, 8:0, 9:7}
    
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
    
    withinclusterdistance = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
    
   #for each cluster
    for i in range(10): 
        for k in range(len(adjustedcluster[i])):
	   withinclusterdistance[i].append(dist.euclidean(adjustedcluster[i][k], cluster_centers[i])) 

    counter = 0
    
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
	if (label == 100): #if the point does not lie within the confidence interval, take the min distance
            label = np.argmin(newdist)
            counter = counter + 1
	finalclusters[i][1] = label
                                    
    with open('submission15_conf_spectral_only.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(finalclusters) 
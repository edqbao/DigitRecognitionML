from sklearn.decomposition import KernelPCA
import csv
import numpy as np
from sklearn.mixture import GaussianMixture

class kPCA_GMM():
    
    # extract_features creates and returns a matrix of the extracted features 
    # of all 10000 images in Extracted_features.csv
    def extract_features(self, csv_file):
        with open(csv_file, 'rb') as csvfile3:
            EF = csv.reader(csvfile3, delimiter=' ', quotechar='|')
            newEF = []
            for row in EF:
                arr = np.asarray(row[0].split(","))
                newEF.append(arr.astype(np.float))
        return newEF
    
    
    # partial_supervision_seed takes as inputs the Seed.csv file and the labels
    # created after performing Gaussian Mixture Model. Seed.csv contains 6 instances
    # of each digit between 0-9. 
    # This functions returns a dictionary where each digit is a key, whose value
    # is the array of the clusters in which the instances of that digit appear in.
    # For example, 
    
    def partial_supervision_seed(self, csv_file, GMM_labels):
        matching = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
        with open(csv_file, 'rb') as csvfile2:
            seedreader = csv.reader(csvfile2, delimiter=' ', quotechar='|')
            for row in seedreader:
                arr = row[0].split(",")
                findClust = GMM_labels[int(arr[0])-1]
                matching[int(arr[1])].append(findClust)
        return matching
    
    
    def assign_to_cluster(self, finalmatches, GMM_labels):
        finalclusters = [[0 for i in range(2)] for j in range(4001)]
        finalclusters[0][0] = 'Id'
        finalclusters[0][1] = 'Label'
    
        for i in range(1,4001):
            finalclusters[i][0] = 6000 + i
            c_label = GMM_labels[5999 + i]
            label = finalmatches.values().index(c_label)
            finalclusters[i][1] = label
        return finalclusters
    
    def createExcel(finalclusters):
        with open('submissionResult.csv', "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(finalclusters)                                  
        
    def main(self):
        algo = kPCA_GMM()
        newEF = algo.extract_features('Extracted_features.csv')
        kpca = KernelPCA(n_components = 30)
        red_pca = kpca.fit_transform(newEF)
        GMM = GaussianMixture(n_components = 10, tol=1e-15, max_iter = 1000, random_state=0)
        GMM = GMM.fit(red_pca)
        GMM_labels = GMM.predict(red_pca)
        matching = algo.partial_supervision_seed('Seed.csv', GMM_labels)
        
        ####  
        for i in range(10):
            print matching[i]
        ####
        
        # Change the line below based on the matching array
        finalmatches = {0:4, 1:1, 2:0, 3:2, 4:6, 5:5, 6:3, 7:7, 8:8, 9:9}
        
        finalclusters =  kPCA_GMM().assign_to_cluster(finalmatches, GMM_labels)
        algo.createExcel(finalclusters)
        
        
if __name__ == '__main__':
    kPCA_GMM().main()
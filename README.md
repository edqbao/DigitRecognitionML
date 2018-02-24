# CS 4786 -- Kaggle Competition
Katerina Prastakou (kmp244), Charlene Luo (cl894), Edward Bao (eqb2)
### Introduction
Our file for running our best predictor, a combination of Kernel PCA and Gaussian Mixture Model can be found in file kPCA_GMM_10000.py. However, the file cannot be run in one piece as there is a section that was done manually. The directions to running our algorithm are as follows:

### Getting Started
- Open the terminal and navigate to the submission folder (Kaggle1), which should include the README.md file and kPCA_GMM_10000.py.
- Move the 3 csv files (Seed.csv, Extracted_features.csv, Graph.csv) into the folder.
- Open kPCA_GMM_10000.py in a text editor.
- Open python from the terminal (on Mac, type “python”).
- Import the 4 packages at the top of the file (import <package>). Then import the file by typing “from kPCA_GMM_10000 import kPCA_GMM”.
- Copy and paste each step in function “main” until the “###” signs into the command line. Up to that line you have reduced the points using kPCA down to 30 dimensions, and performed GMM to all the points.
- Copy and paste the text between the “####” signs. 
```sh
> for i in range(10):
    print (matching[i])
```
- This will output 10 lines, one for each digit between 0-9 with an array of 6 numbers -- the clusters corresponding to the 6 instances of that digit in seed.csv. For example, if the first line is the array [2,2,2,2,2,2] that means the 6 0’s were all placed in cluster 2. If the second line is [0,0,7,0,0,0] that means 5 1’s were placed in cluster 0, and one was placed in cluster 7, and so on. The following step is critical to the success of the program.
- In the line “finalmatches = {0:4, 1:1, 2:0, 3:2, 4:6, 5:5, 6:3, 7:7, 8:8, 9:9}”, the values of the dictionary have to be changed depending on the results of GMM (which are different each time because of a random parameter). This step matches each cluster to a specific digit, and thus is very important in the success of the algorithm. In the “finalmatches” dictionary, the keys correspond to the digits, while the values correspond to their clusters. The cluster to be assigned to each digit is the mode of the corresponding line. For example, if line 4 (digit 3) results in [4,4,4,4,3,3] then we would assign cluster 4 to digit 3 by setting the value of 3 to 4 (3:4). However, each cluster has to uniquely be assigned to a digit. Thus, if the mode of digits 1 and 2 are both cluster 3, but 5 1’s were placed in cluster 3, while only 3 2’s were placed in the cluster, then 3 will be assigned to 1 and the next most frequented will be assigned to 2. 

#### Example of Cluster Assignment
```diff
[0, 0, 0, 0, 0, 0] --> 0
[9, 3, 9, 9, 9, 9] --> 9
[1, 1, 1, 1, 1, 1] --> 1
[4, 4, 4, 4, 6, 4] --> 4
[2, 5, 5, 5, 5, 2] --> 5
[4, 5, 6, 6, 6, 6] --> 6
[7, 7, 7, 7, 7, 0] --> 7
[5, 3, 5, 2, 8, 8] --> 8
- [6, 4, 4, 6, 4, 1] --> 3
[5, 8, 2, 2, 2, 8] --> 2
```

- Notice on the second to last line, cluster 3 is assigned even though it does not appear in the array. This might happen, as there are more instances of 6’s, 4’s and 1’s in other digits than in digit 8. In that case, we assign the last remaining cluster, which was 3. The final dictionary would be: 
```sh
> finalmatches = {0:0, 1:9, 2:1, 3:4, 4:5, 5:6, 6:7, 7:8, 8:3, 9:2}”
```
- After updating final matches, copy and paste the last 2 lines from the main function. The prediction results will be extracted into a csv file called submissionResult.csv.


### Requirements
Python 2.7+ installed
Packages: sklearn.decomposition, csv, numpy, sklearn.mixture


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>

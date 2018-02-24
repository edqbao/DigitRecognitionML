import numpy as np;
import scipy.sparse.linalg as linalg;
def PCA(X,k):
    sigma = np.cov(X);
    e, w = linalg.eig(sigma,k,which='LM');
    w=np.transpose(w);
    mean=np.matlib.repmat(np.mean(X),50,1000);
    return np.matmul(np.transpose((X-mean)),w);
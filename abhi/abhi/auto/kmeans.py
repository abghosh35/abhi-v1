from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Clustering:
    """
    Build CLUSTERS using KMeans algorithm
    Inputs: X (a pandas dataframe of features based on which clustering will be done)
    Parameters:
        use_pca: Boolean (default=True) Used to determine if Principal Component analysis will be done. This is useful for visualizing the clusters in a 2-Dimensional space. If use_pca=False and dim(X) > 2, then plots will not be returned.
        auto_fit: Boolean (default=True) If true Silhoutte score will be used to determine the best fitted clusters.
        random_state: Int (default=453) It is best practise to set a random seed to reproduce results
    """
    def __init__(self, X=None, min_clusters=5, max_clusters=10, use_pca=True, auto_fit=True, random_state=453):
        self.X = X
        self.use_pca = use_pca
        self.auto_fit = auto_fit
        self.random_state = random_state
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
    def fitpca(self):
        princomp = PCA(n_components=2, random_state=self.random_state)
        princomp.fit(self.X)
        
        return princomp.transform(self.X)
    
    
    def silhoute_score(self):
        sil_score = []
        krange = np.arange(self.min_clusters, self.max_clusters)
        
        if self.use_pca:
            X = self.fitpca()
        else:
            X = self.X

        for k in krange:
            clust = KMeans(n_clusters=k, random_state=self.random_state, n_jobs=-1, max_iter=1e4)
            clust.fit(X)

            sil_score.append(silhouette_score(X, clust.labels_, metric="euclidean"))

        return krange, sil_score
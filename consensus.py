from cfa import factor
import numpy as np
import sklearn as sk
import scipy as sp
from sklearn import cluster
from tqdm import tqdm

class ConsensusFactorModel:
    def __init__(self, k, n_reps = 10, random_states = None, model_params = {}):
        self.k = k
        self.n_reps = n_reps
        self.random_states = list(range(self.n_reps)) if random_states is None else random_states
        self.models = [factor.LinearFactorModel(self.k, random_state = s, **model_params) for s in self.random_states]
        self.factor_keys = ["Ve", "Vm", "Vb_0", "Vb_1"]
    def fit(self, data_dict):
        for m in tqdm(self.models):
            m.fit(data_dict)
    def aggregate_factors(self):
        def l2_norm(x):
            return x / np.linalg.norm(x, axis = 0)
        self.factor_run_id = np.hstack([np.full(m.k, i) for (i, m) in enumerate(self.models)])
        self.factor_aggregate = np.hstack([np.vstack([l2_norm(m.factors[x]) for x in self.factor_keys]) for m in self.models])
    def detect_outliers(self, n_neigh = None, q = 0.9):
        factors = self.factor_aggregate
        n_neigh = int(0.5*self.k) if n_neigh is None else n_neigh
        self.factor_graph = sk.neighbors.kneighbors_graph(factors.T, n_neighbors=n_neigh)
        self.mean_dist_to_k = np.array([sp.spatial.distance.cdist(factors[:, [i, ]].T, factors[:, np.where(np.asarray(self.factor_graph[i, :].todense()).flatten())[0]].T).mean() for i in range(self.factor_graph.shape[0])])
        self.is_outlier = self.mean_dist_to_k > np.quantile(self.mean_dist_to_k, q)
    def get_consensus_factors(self, outlier_params = {}):
        self.aggregate_factors()
        self.detect_outliers(**outlier_params)
        factors = self.factor_aggregate
        self.kmeans_op = sk.cluster.KMeans(n_clusters=self.k).fit(factors[:, ~self.is_outlier].T)
        self.factor_clusts = self.kmeans_op.predict(factors.T)
        self.factor_clusts[self.is_outlier] = -1
        factors_raw = {x : np.hstack([m.factors[x] for m in self.models]) for x in self.factor_keys}
        clust_idxs = [np.where(self.factor_clusts == i)[0] for i in range(self.k)]
        self.cons_factors = [{k : np.median(factors_raw[k][:, i], 1) for k in self.factor_keys} for i in clust_idxs]

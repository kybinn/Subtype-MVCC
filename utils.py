import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import time

cuda = True if torch.cuda.is_available() else False

def calculate_similarity(data):
    X=data
    dot_products = np.dot(X, X.T)
    norm_squared = np.linalg.norm(X, axis=1)**2
    norm_products = np.outer(norm_squared, norm_squared)
    return dot_products / norm_products

    similarity_matrix = 1.0 / (1.0 + distance_matrix)

    return similarity_matrix


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def graph_from_dist_tensor(dist, num_class, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    
    k=int(dist.shape[0]/num_class+1)  
    new_matrix = np.ones_like(dist)
    for idx, row in enumerate(dist):
        kth_smallest = np.partition(row, k-1)[:k]
        new_row = np.where(np.isin(row, kth_smallest), row, 1)
        new_matrix[idx] = new_row

    return new_matrix

def gen_adj_mat_tensor(data, num_class, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)  
    g = graph_from_dist_tensor(dist, num_class, self_dist=True) 

    if metric == "cosine":
        # H=data
        # H_norm = H / H.norm(dim=1, keepdim=True)
        # adj = torch.mm(H_norm, H_norm.t())

        adj = 1-g    
    else:
        raise NotImplementedError
    


    diag_idx = np.diag_indices(adj.shape[0])
    adj[diag_idx[0], diag_idx[1]] = 0
    


    row_sums = adj.sum(axis=1)                          
    row_sums[row_sums==0]=1 

    row_sums_expanded = np.expand_dims(row_sums, axis=1)
    adj = adj / row_sums_expanded        

    adj_T=adj.T
    adj=adj+adj_T                      
    adj = F.normalize(torch.from_numpy(adj), p=1)  
    I = torch.eye(adj.shape[0])
    adj=adj+I
    adj = to_sparse(adj)
    
    return adj



def knbrsloss(H, k, eps=1e-8):
    
   
    H_norm = H.norm(dim=1, keepdim=True)
    dist_matrix = torch.mm(H, H.t())/(H_norm * H_norm.t()).clamp(min=eps)


   
    simMaxNeb, indices = torch.topk(dist_matrix, k, largest=True)
  
    indices = indices[:, 1:]
    simMaxNeb = simMaxNeb[:, 1:]

    f = lambda x: torch.exp(x)
    refl_sim = f(dist_matrix)
   
    
    num = H.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    V = torch.zeros((num, k-1)).to(device)
    
    
    V = f(simMaxNeb) 

    ret = -torch.log(
        V.sum(1) / (refl_sim.sum(1) - refl_sim.diag()))
       
    ret = ret.mean()

    return ret


def cluster(method,data_matrix,num_class):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_class)
        kmeans.fit(data_matrix)
        labels = torch.tensor(kmeans.labels_)
    elif method == 'spectral':
        spectral = SpectralClustering(n_clusters=num_class)
        labels = torch.tensor(spectral.fit_predict(data_matrix))
    elif method == 'agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=num_class)
        labels = torch.tensor(agglomerative.fit_predict(data_matrix))
    elif method == 'gmm':
        gmm = GaussianMixture(n_components=num_class)
        gmm.fit(data_matrix)
        labels = torch.tensor(gmm.predict(data_matrix))
    else:
        raise ValueError("Invalid clustering method name. Available options are 'kmeans', 'spectral', 'agglomerative', 'gmm'.")
    
    return labels




def float2lable(result):
    max_indices = np.argmax(result, axis=1)
    predicted_labels = max_indices+1
    return predicted_labels

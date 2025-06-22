import argparse
import pandas as pd

from sklearn.metrics import silhouette_score

from subtype_mvcc.train import train_test
from subtype_mvcc.utils import cluster
from subtype_mvcc.consensus_clustering import ConsensusCluster

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    
    parser = argparse.ArgumentParser(description='SubtypeDCGCN v1.0')
    parser.add_argument('--input_files', type=str, required=True, help='Comma-separated paths to omics input files')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder')
    parser.add_argument("--clusters ", dest='cluster_num', type=str, default=-1, help="cluster number")
    parser.add_argument("--seed ", dest='seed', type=int, default=42, help="Seed for clustering")
    
    num_epoch =600 
    lr_e = 1e-4
    #lr_e = 5e-4
    
    args = parser.parse_args()
    set_seed(args.seed)

    input_files = args.input_files.split(',')
    output_folder = args.output_folder
    
    view_list = [i for i in range(len(input_files))] 
    mean_emb=[]
    labels_dict={} 
    
    if args.cluster_num == "auto": 
        clusters_to_test_min = 2
        clusters_to_test_max = 9
        cluster_n = 6 # Default cluster number, required to train_test as the cluster number is required in the to pick other parameters
    else:
        cluster_n = int(args.cluster_num)

    mean_emb,index=train_test(input_files, output_folder, view_list, cluster_n,lr_e,num_epoch)   
    mean_emb = pd.DataFrame(data=mean_emb.detach().cpu().numpy(), index=index, columns=map(lambda x: 'v' + str(x), range(mean_emb.shape[1])))
    #fea_tmp_file = output_folder + "/features.fea"
    #fea.to_csv(fea_tmp_file, header=True, index=True, sep='\t')

    #mean_emb= pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
    
    if args.cluster_num == "auto":
        # Perform consensus clustering to determine the optimal number of clusters
        consensus = ConsensusCluster(cluster, L=clusters_to_test_min, K=clusters_to_test_max+1, H=10, resample_proportion=0.8)
        consensus.fit(mean_emb.values)
        cluster_n = consensus.bestK
        print("Optimal number of clusters:", cluster_n)

    method=['kmeans']
    for choice in method:
        labels=cluster(method=choice,data_matrix=mean_emb,num_class=cluster_n, seed = args.seed)+1
        labels_dict[choice]=labels 
        mean_emb[choice]=labels  
        X = mean_emb.loc[:, [choice]] 
        out_file = output_folder + '/result.dcgcn'
        X.to_csv(out_file, header=True, index=True, sep='\t')

    for method, labels in labels_dict.items():
        score = silhouette_score(mean_emb, labels)
        print('Method: {}, silhouette score: {:.2f}'.format(method, score))

        
if __name__ == "__main__":
    main()

import argparse
import sys
import numpy as np
import random
import time
import os
import pandas as pd
from os.path import isfile,splitext,basename
from .train import train_test,train_test_extension
from .utils import cluster,calculate_similarity,float2lable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering

def main(argv=sys.argv):
    
    parser = argparse.ArgumentParser(description='SubtypeDCGCN v1.0')
    parser.add_argument("-i", dest='file_input', default="./input/input.list",
                        help="file input")
    parser.add_argument("-m", dest='run_mode', default="feature", help= "run_mode: feature, cluster")
    parser.add_argument("-n", dest='cluster_num', type=int, default=-1, help="cluster number")
    parser.add_argument("-t", dest='type', default="ALL", help="cancer type: BRCA, GBM")

    
    view_list = [1,2,3,4]
    
    num_epoch =600 
    lr_e = 1e-4
    #lr_e = 5e-4
    
    time_start = time.time()
    args = parser.parse_args()
    cancer_type = args.type
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                    'GBM': 3, 'LUAD': 3, 'PAAD': 2,
                    'SKCM': 4, 'STAD': 3, 'UCEC': 4, 'UVM': 4}
    mean_emb=[]
    labels_dict={} 
    fea_tmp_file = './fea/' + cancer_type + '.fea' 

    if cancer_type not in cancer_dict and args.cluster_num == -1:
            print("Please set the number of clusters!")
    elif args.cluster_num == -1:
            args.cluster_num = cancer_dict[cancer_type]
    
    print("python main.py -t"+cancer_type)
            
    # python main.py -t KIRC -i ./input/KIRC.list -m tsne
    # python main.py -t BRCA -i ./input/BRCA.list -m training 
    # python main.py -t PAAD -i ./input/PAAD.list -m training 
    # python main.py -t BLCA -i ./input/BLCA.list -m training
    # python main.py -t UVM -i ./input/UVM.list -m training
    if args.run_mode == 'training':
        mean_emb,index=train_test(args.file_input,cancer_type, view_list, args.cluster_num,lr_e,num_epoch)   
        # mean_emb,index=train_test_extension(args.file_input,cancer_type, view_list, args.cluster_num,lr_e,num_epoch)              
        
        fea = pd.DataFrame(data=mean_emb.detach().cpu().numpy(), index=index, columns=map(lambda x: 'v' + str(x), range(mean_emb.shape[1])))
        fea.to_csv(fea_tmp_file, header=True, index=True, sep='\t')
        time_end2 = time.time()
        time_sum = time_end2 - time_start
        print(time_sum)

    
    elif args.run_mode == 'clustering':
        if isfile(fea_tmp_file):
            mean_emb= pd.read_csv(fea_tmp_file, header=0, index_col=0, sep='\t')
            method=['kmeans']
            for choice in method:
                labels=cluster(method=choice,data_matrix=mean_emb,num_class=args.cluster_num)+1
                labels_dict[choice]=labels 
      
                mean_emb[choice]=labels  
                X = mean_emb.loc[:, [choice]] 
                out_file = './analysis/results/' + cancer_type + '.dcgcn'
                X.to_csv(out_file, header=True, index=True, sep='\t')

            for method, labels in labels_dict.items():
                score = silhouette_score(mean_emb, labels)
                print('Method: {}, silhouette score: {:.2f}'.format(method, score))
        else:
            print('file does not exist!')

        
if __name__ == "__main__":
    main()

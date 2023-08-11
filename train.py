import os
import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import init_model_dict, init_optim
from utils import  gen_adj_mat_tensor,knbrsloss
from os.path import splitext, basename, isfile


cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(file_input,cancer_type): 
    l = [] 
    data_tr_list = []
    tmp_dir = './fea/' + cancer_type + '/'
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    
    for line in open(file_input, 'rt'): 
        base_file = splitext(basename(line.rstrip()))[0]
        fea_save_file = tmp_dir + base_file + '.fea'
        if isfile(fea_save_file):
            df_new = pd.read_csv(fea_save_file, sep=',', header=0, index_col=0)    
            l = list(df_new)         
        df_new = df_new.T
        data_tr_list.append(df_new.values.astype(float))
    data_tensor_list = []
    for i in range(len(data_tr_list)): 
        data_tensor_list.append(torch.FloatTensor(data_tr_list[i]))
 
    return data_tensor_list,l


def gen_trte_adj_mat(data_tr_list,num_class):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    for i in range(len(data_tr_list)): 
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], num_class, adj_metric)) #得到当前组学的相似度矩阵
        
    
    return adj_train_list




def train_epoch(data_list, adj_list, model_dict, optim_dict,k,pre_flag):

    torch.autograd.set_detect_anomaly(True)
    loss_dict = {}
    embedding_list=[]  # embeddings list
    output_list1=[] 
    output_list2=[] 
    mean_emb = []

    criterion = torch.nn.MSELoss()# MSE
    for m in model_dict:
        model_dict[m].train()  
    num_view = len(data_list)

    
    if pre_flag is True:
        ci_loss = 0     
        gi_loss = 0     
        for i in range(num_view):
            optim_dict["G{:}".format(i+1)].zero_grad()  
            embedding_list.append(model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
           
        
        stack_tensor_list2 =torch.stack(embedding_list)
        mean_clu = stack_tensor_list2.mean(dim=0)
        for i in range(num_view):
            ci_loss= criterion(embedding_list[i], torch.nn.Parameter(mean_clu, requires_grad=False)) 
            gi_loss=knbrsloss(embedding_list[i], k)
            total_ccloss=ci_loss+gi_loss
            total_ccloss.backward()             
            
            optim_dict["G{:}".format(i+1)].step()
            loss_dict["G{:}".format(i+1)] = gi_loss.detach().cpu().numpy().item()
        

        
        
    if num_view >= 2:
        m_loss = 0
        for i in range(num_view):
            optim_dict["M{:}".format(i+1)].zero_grad()
            embedding_list[i]=model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])
        
        stack_tensor_list =torch.stack(embedding_list)
        mean_emb = torch.mean(stack_tensor_list,dim=0) 
       
        
        
        for i in range(num_view):
            
            output_list2.append(model_dict["M{:}".format(i+1)](torch.nn.Parameter(mean_emb, requires_grad=False)))
            m_loss = criterion(output_list2[i], data_list[i])
            m_loss.backward(retain_graph=True)
            optim_dict["M{:}".format(i+1)].step()
            loss_dict["M{:}".format(i+1)] = m_loss.detach().cpu().numpy().item()

    return loss_dict,embedding_list,mean_emb
    
       
def train_test(file_input,data_folder, view_list, num_class,lr_e, lr_pre,pre_epoch):
    num_view = len(view_list)           
    dim_he_list = [300,200,100]         
    l=[]
    ############## prepare_trte_data ###################
    data_tr_list,l = prepare_trte_data(file_input,data_folder) 
    

    ############## gen_trte_adj_mat ####################
    adj_tr_list = gen_trte_adj_mat(data_tr_list,num_class)
    k=int(data_tr_list[0].shape[0]/num_class+1)  

    dim_list = [x.shape[1] for x in data_tr_list] 


    ############## init_model_dict ####################
    model_dict = init_model_dict(num_view, dim_list, dim_he_list,num_class)
    
    
    embedding_list=[] 
    loss_dict={} 
    mean_emb=[]  
    loss_list_dict={} 
    optim_dict = init_optim(num_view, model_dict, lr_e)
    
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    for m in range(num_view):
        if cuda:
            data_tr_list[m]=data_tr_list[m].cuda()
            adj_tr_list[m]=adj_tr_list[m].cuda()

    print("\ntrain ...\n")
    loss_list_dict={}
    
    optim_dict1 = init_optim(num_view, model_dict, lr_pre) 
    for epoch in range(pre_epoch+1): 
        loss_dict,embedding_list,mean_emb=train_epoch(data_tr_list, adj_tr_list, model_dict, optim_dict1,k,True)
        for key in loss_dict:
            loss_list_dict.setdefault(key, []).append(loss_dict[key]) 
            
        if(epoch==pre_epoch):
            print("Epoch {}: loss_dict = {}".format(epoch, loss_dict))

    for key in loss_list_dict:
        plt.plot(loss_list_dict[key])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        # plt.show()    
        plt.savefig('./loss_png/'+data_folder+('{}_show.png'.format(key)))
    
    
    return mean_emb,l


def train_epoch_extension(data_list, adj_list, model_dict, optim_dict,k,keep,index,pre_flag):
    torch.autograd.set_detect_anomaly(True)
    loss_dict = {}  
    embedding_list=[]  # embeddings list
    output_list1=[] # output list1 embeddings
    output_list2=[] # output list mean_emb
    mean_emb = []

    criterion = torch.nn.MSELoss()# MSE
    for m in model_dict:
        model_dict[m].train()  
    num_view = len(data_list)

    
    if pre_flag is True:
        ci_loss = 0     
        gi_loss = 0    
        for i in range(num_view):
            optim_dict["G{:}".format(i+1)].zero_grad()  
            embedding_list.append(model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        
        stack_tensor_list2 =torch.stack(embedding_list)
       
        if(cuda):
            mean_clu = torch.sum(stack_tensor_list2,dim=0)/torch.tensor(keep).unsqueeze(1).repeat(1, 100).cuda() 
        else:
            mean_clu = torch.sum(stack_tensor_list2,dim=0)/torch.tensor(keep).unsqueeze(1).repeat(1, 100)
        
        for i in range(num_view):
            ci_loss= criterion(embedding_list[i][np.where(np.array(index[i]) == 1)[0], :], torch.nn.Parameter(mean_clu[np.where(np.array(index[i]) == 1)[0], :], requires_grad=False)) 
            gi_loss=knbrsloss(embedding_list[i], k)
           
            total_ccloss=ci_loss+gi_loss
            total_ccloss.backward()             
            optim_dict["G{:}".format(i+1)].step()
            loss_dict["G{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
        

    
    if num_view >= 2:
        m_loss = 0
        for i in range(num_view):
            optim_dict["M{:}".format(i+1)].zero_grad()
            embedding_list[i]=model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])
        
        stack_tensor_list =torch.stack(embedding_list)
        if cuda:
            mean_emb = torch.sum(stack_tensor_list,dim=0)/torch.tensor(keep).unsqueeze(1).repeat(1, 100).cuda()
        else:
            mean_emb = torch.sum(stack_tensor_list,dim=0)/torch.tensor(keep).unsqueeze(1).repeat(1, 100)
        
        for i in range(num_view):
            output_list2.append(model_dict["M{:}".format(i+1)](torch.nn.Parameter(mean_emb, requires_grad=False)))
           
            m_loss = criterion(output_list2[i][np.where(np.array(index[i]) == 1)[0], :], data_list[i][np.where(np.array(index[i]) == 1)[0], :])
            m_loss.backward(retain_graph=True)
            optim_dict["M{:}".format(i+1)].step()
            loss_dict["M{:}".format(i+1)] = m_loss.detach().cpu().numpy().item()

    return loss_dict,embedding_list,mean_emb
    

def train_test_extension(file_input,data_folder, view_list, num_class,lr_e, lr_pre,pre_epoch):
    num_view = len(view_list)          
    dim_he_list = [300,200,100]       
    l=[]
    ############## prepare_trte_data ###################
    data_tr_list,l = prepare_trte_data(file_input,data_folder) 
    array_length = data_tr_list[0].shape[0]
    

    lost_lst =[None]*array_length
    num_zeros = int(array_length * 0.9) 
    num_ones = int(array_length * 0.03)
    num_twos = int(array_length * 0.03)
    num_threes = array_length-num_ones-num_zeros-num_twos


    lost_lst[:num_zeros-1]=[0]*num_zeros
    lost_lst[num_zeros:num_zeros+num_ones-1]=[1]*num_ones
    lost_lst[num_zeros+num_ones:num_zeros+num_ones+num_twos-1]=[2]*num_twos
    lost_lst[num_zeros+num_ones+num_twos:]=[3]*num_threes
    
    random.shuffle(lost_lst)
    keep_list = [4 - x for x in lost_lst]
    index_list = [[1] * array_length for _ in range(num_view)] 
    for i in range(array_length):
        de_num = lost_lst[i] 
        if de_num!=0:
            delete_view_list=random.sample(range(num_view),de_num)
            for j in delete_view_list:
                data_tr_list[j][i]=0   
                index_list[j][i]=0     





    ############## gen_trte_adj_mat ####################
    adj_tr_list = gen_trte_adj_mat(data_tr_list,num_class)
    k=int(data_tr_list[0].shape[0]/num_class+1)  
    dim_list = [x.shape[1] for x in data_tr_list] 
    ############## init_model_dict ####################
    model_dict = init_model_dict(num_view, dim_list, dim_he_list,num_class)
    
    
    embedding_list=[]
    loss_dict={} 
    mean_emb=[]  
    loss_list_dict={} 
    optim_dict = init_optim(num_view, model_dict, lr_e)
    
    ################ GPU ############################
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    for m in range(num_view):
        if cuda:
            data_tr_list[m]=data_tr_list[m].cuda()
            adj_tr_list[m]=adj_tr_list[m].cuda()
   
    ############## Pretrain #########################
   
    print("\ntrain ...\n")
    loss_list_dict={}

    optim_dict1 = init_optim(num_view, model_dict, lr_pre) 
    for epoch in range(pre_epoch+1):
        loss_dict,embedding_list,mean_emb=train_epoch_extension(data_tr_list, adj_tr_list, model_dict, optim_dict1,k,keep_list,index_list,True)
        for key in loss_dict:
            loss_list_dict.setdefault(key, []).append(loss_dict[key]) 
            
        if(epoch==pre_epoch):
            print("Epoch {}: loss_dict = {}".format(epoch, loss_dict))

    for key in loss_list_dict:
        plt.plot(loss_list_dict[key])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        # plt.show()    
        plt.savefig('./loss_png/'+data_folder+('{}_show.png'.format(key)))
    
    
   
    return mean_emb,l

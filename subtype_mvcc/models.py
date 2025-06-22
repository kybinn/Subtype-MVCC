""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)) 
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
            
    def forward(self, x, adj):
        support = torch.mm(x, self.weight) 
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    

class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj) 
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        
        return x


class decoder_D(nn.Module):
    def __init__(self, in_dim,hid_dim, out_dim):
        super(decoder_D, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out 



def init_model_dict(num_view, dim_list, dim_he_list,num_class, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["M{:}".format(i+1)] = decoder_D(dim_he_list[-1],200,dim_list[i]) 

    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4):
    optim_dict = {}
    
    for i in range(num_view):
        optim_dict["G{:}".format(i+1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i+1)].parameters()),
            lr=lr_e) 
        optim_dict["M{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["M{:}".format(i+1)].parameters()), 
                lr=lr_e)

   
    return optim_dict

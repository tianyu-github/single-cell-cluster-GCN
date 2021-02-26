#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:06:03 2021

@author: tianyu
"""

import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
import sys
sys.path.insert(0, 'lib/')

from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter
import utilsdata
from utilsdata import *
# torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)
        self.fc1 = nn.Linear(n_z, 500)
        self.fc2 = nn.Linear(500, n_input)
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
            
        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = 0,0,0,0,0  #self.ae(x)
        
        sigma = 0

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj, active=False)
        h5 = self.gnn_5((1-sigma)* F.relu(h) + sigma*z, adj, active=False)
        predict = F.softmax(h5, dim=1)
        
        
        deco = self.fc1(F.relu(h))
        x_bar = self.fc2(F.relu(deco))
        x_bar = F.relu(x_bar)
        
        adj_bar = 0 #F.relu(torch.mm(h, h.t()))
        
        #print(predict.shape, h.shape, h.shape, h.unsqueeze(1).shape, self.cluster_layer.shape)
        #print(self.cluster_layer)
        #print("+++",(h.unsqueeze(1) - self.cluster_layer).shape)
        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        #print("---",q.shape)   
        
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, adj_bar


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()





    
parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--user', type=str, default='personal', help="personal or hpc")
parser.add_argument('--name', type=str, default='kolod')
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--lr', type=float, default= 0.001 )
parser.add_argument('--n_clusters', default=4, type=int)
parser.add_argument('--n_z', default=10, type=int)
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')

#parser.add_argument('--pretrain_path', type=str, default='pkl')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

#args.pretrain_path = 'data/{}.pkl'.format(args.name)
#

if args.name == 'usoskin':
    
    filepath = '/Users/tianyu/google drive/fasttext/imputation/' if args.user == 'personal' else '/gpfs/sharedfs1/shn15004/tianyu/'    
    adj, features,labels = utilsdata.load_usoskin(path = filepath, dataset='usoskin')
    print(adj.shape, features.shape, len(labels))
    features, geneind = utilsdata.high_var_npdata(features, num=args.num_gene, ind=1)
    
    features = features/np.max(features)
    
    print('******************************',adj.shape, features.shape)
    
    args.n_clusters = 4
    args.n_input = args.num_gene
    dataset = load_data(features.T, labels)

if args.name == 'kolod':
    
    filepath = '/Users/tianyu/google drive/fasttext/imputation/' if args.user == 'personal' else '/gpfs/sharedfs1/shn15004/tianyu/'    
    adj, features,labels = utilsdata.load_kolod(path = filepath, dataset='kolod')
    print(adj.shape, features.shape, len(labels))
    features, geneind = utilsdata.high_var_npdata(features, num=args.num_gene, ind=1)
    
    features = features/np.max(features)
    
    print('******************************',adj.shape, features.shape)
    
    args.n_clusters = 3
    args.n_input = args.num_gene
    dataset = load_data(features.T, labels)    
    
print(args)

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph

def buildGraphNN(X):
    #nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    #distances, indices = nbrs.kneighbors(X)
    A = kneighbors_graph(X, 5, mode='connectivity', include_self=True)
    return A


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
#            torch.nn.init.xavier_uniform_(m.bias)
            m.bias.data.fill_(0.0)
def pretrain_ae(dataset):
    model_ae = AE(256, 128, 64, 64, 128, 256,
                n_input=args.n_input,
                n_z=args.n_z,
                ).to(device)
    print(model_ae)
    optimizer = Adam(model_ae.parameters(), lr=args.lr)
    
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    for epoch in range(500):
        if epoch % 50 == 0:
        # update_interval
            x_bar, enc_h1, enc_h2, enc_h3, z = model_ae(data)

        x_bar, enc_h1, enc_h2, enc_h3, z = model_ae(data)
  
        loss = F.mse_loss(x_bar, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

def train_sdcn(dataset):
    model = SDCN(256, 128, 64, 64, 128, 256,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)
    # instantiate the object net of the class
    #model.apply(weight_init)


    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
#    adj = load_graph(args.name, args.k)
#    adj = adj.cuda()
    adj = buildGraphNN(features.T)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

#    with torch.no_grad():
#        _, _, _, _, z = model.ae(data)    
#    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
#    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features.T)
    y_pred_last = y_pred
    ######model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'kmeans')


            

    for epoch in range(100):
        if epoch % 20 == 0:
        # update_interval
            _, tmp_q, pred, _,_ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, _, adj_bar = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        #reAJD_loss = F.mse_loss(adj_bar, adj)
        #print(kl_loss, ce_loss, re_loss)
        loss = 0.01 * kl_loss + 0.1 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
train_sdcn(dataset)
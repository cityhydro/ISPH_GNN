#!usr/env/bin python3
import torch
import numpy as np
import joblib
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import argparse
from sklearn.preprocessing import StandardScaler as Scaler
import pickle
import logging

from nn_module_new import NodeNetwork, EdgeNetwork, LIN_KERNEL
from Constants import LAP_RADIUS, COL_RADIUS
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


class bxcg(object):
    dp={}
    
    def __init__(self, step):
        self.step = step
        self.m= step


    def ab(self):
        ap={}
        
        if self.step == 231:
            break_point_pt = 'training/prs_net_small_state/ckpt_final.pkl'
            
            with open(break_point_pt, 'rb') as pickle_file:
                why = pickle.load(pickle_file)
            self.dp['model_state']= why['model_state']
            self.dp['vel_scaler']= why['vel_scaler']
            self.dp['div_scaler']= why['dns_scaler']
            self.dp['pre_scaler']= why['pre_scaler']
            self.dp['prs_scaler']= why['prs_scaler']
        
            #self.dp['pos']= 1
        
        
        ap=self.dp

        return ap

class zcg(object):
    dp={}
    prs_net = NodeNetwork(4, 1, 2, 0.55*LAP_RADIUS)
    
    def __init__(self, step):
        self.step = step
        self.m= step


    def abc(self):
        
        
        if self.step == 231:
            break_point_pt = 'training/prs_net_small_state/ckpt_final.pkl'
            
            with open(break_point_pt, 'rb') as pickle_file:
                why = pickle.load(pickle_file)
            self.dp['model_state']= why['model_state']
            self.dp['vel_scaler']= why['vel_scaler']
            self.dp['div_scaler']= why['dns_scaler']
            self.dp['pre_scaler']= why['pre_scaler']
            self.dp['prs_scaler']= why['prs_scaler']
            self.prs_net.eval()
            self.prs_net.load_state_dict(self.dp['model_state'])
            
 
            self.prs_net.cuda()
            #self.dp['pos']= 1
        
        
        ap=self.prs_net
        return ap


class input(object):
    #inipos = {}
    #inivel = {}
    #inidns = {}
    #inipre = {}
    dp={}
    #print (dp)

    
    def __init__(self, step, prs_feat):
        self.step = step
        self.m= step
        self.prs_feat=prs_feat

    def abcd(self):
        
        if self.step == 231:
            self.dp['inipos']= torch.from_numpy(self.prs_feat['pos']).cuda()
            self.dp['inivel']= torch.from_numpy(self.prs_feat['vel']).cuda()
            self.dp['inidns']= torch.from_numpy(self.prs_feat['dns']).cuda()
            self.dp['inidiv']= torch.from_numpy(self.prs_feat['div']).cuda()
            self.dp['inifs']= torch.from_numpy(self.prs_feat['fs']).cuda()
            self.dp['inipre']= torch.from_numpy(self.prs_feat['pre']).cuda()
            
            #self.inipos = torch.from_numpy(self.prs_feat['pos']).cuda()
            #self.inivel = torch.from_numpy(self.prs_feat['vel']).cuda()
            #self.inidns = torch.from_numpy(self.prs_feat['dns']).cuda()
            #self.inipre = torch.from_numpy(self.prs_feat['pre']).cuda()
            

            #self.dp[self.inipos.cuda()'pos']= 1
        
        
        ap=self.dp

        #bp=self.inivel
        #cp=self.inidns
        #dp=self.inipre
        #print(ap)
        return ap  



def run_sim(step,num,fpos,fvel,fdns,fdiv,fs,fpre):

    #print('Viscosity:%.6f' % case.nu)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('using device ' + device)

    cp=bxcg(step)
    bp=cp.ab()
    #print(bp['model_state'])

    

    pnet=zcg(step)
    prs_net=pnet.abc()
    model=pnet.abc()
    #
    #print ("ING20")
    
    prs_vel_norm_scaler = bp['vel_scaler']
    prs_div_scaler = bp['div_scaler']
    prs_pre_scaler = bp['pre_scaler']
    prs_scaler = bp['prs_scaler']
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        
    #prs_feat = {
    #        'pos': pos,
    #        'dns': dns[:,0],
    #        'vel': vel}
        prs_feat ={}
        prs_feat['pos']=fpos.astype(np.float32)
        prs_feat['dns']=fdns[:,0].astype(np.float32)
        prs_feat['div']=fdiv[:,0].astype(np.float32)
        prs_feat['fs']=fs[:,0].astype(np.float32)
        prs_feat['pre']=fpre[:,0].astype(np.float32)
        prs_feat['vel']=fvel.astype(np.float32)
        
        

        vel_norm_scaler = Scaler()
        dns_scaler = Scaler()
        div_scaler = Scaler()
        pre_scaler = Scaler()
    #prs_scaler = Scaler()
    
        incuda=input(step,prs_feat)
        ipos=incuda.abcd()
    #ipos['inipos']=ipos['inipos']+torch.from_numpy(prs_feat['pos'])
    #print(ipos)

        pos=ipos['inipos']
        vel=ipos['inivel']
        dns=ipos['inidns']
        div=ipos['inidiv']
        fs=ipos['inifs']
        pre=ipos['inipre']
        #print(vel.shape)
        
        #print(dns)
        cpos = torch.from_numpy(prs_feat['pos'])
        cvel = torch.from_numpy(prs_feat['vel'])
        cdns = torch.from_numpy(prs_feat['dns'])
        cdiv = torch.from_numpy(prs_feat['div'])
        cfs = torch.from_numpy(prs_feat['fs'])
        cpre = torch.from_numpy(prs_feat['pre'])
        #print(dns)
        pos[:,:]=cpos[:,:]
        vel[:,:]=cvel[:,:]
        dns[:]=cdns[:]
        div[:]=cdiv[:]
        fs[:]=cfs[:]
        pre[:]=cpre[:]
        #print(dns

        # normalize vel
        velin= torch.mul(vel,vel)
        veladd=torch.sqrt(velin[:,0]+velin[:,1])
        vel_norm = veladd.view(-1, 1) + 1e-8
        #vel_norm = vel[:,1] / np.sqrt(velin)
        #vel_norm = vel_norm.view(-1, 1) + 1e-8

        vel_norm_scaler.partial_fit(vel_norm.cpu().numpy())
        #vel_norm = (vel_norm - vel_norm_scaler.mean_.item()) / np.sqrt(vel_norm_scaler.var_.item())
        vel_norm = vel_norm / np.sqrt(vel_norm_scaler.var_.item())
        #vel = vel.to(device)
        #vel_norm=vel_norm.to(device)
        #vel = vel / np.sqrt(vel_norm_scaler.var_.item())
        vel = (vel - prs_vel_norm_scaler.mean_.item()) / np.sqrt(prs_vel_norm_scaler.var_.item())
        #vel = vel / np.sqrt(prs_vel_norm_scaler.var_.item())

        #dns = (dns - dns_scaler.mean_.item()) / np.sqrt(dns_scaler.var_.item())
        #dns = dns / np.sqrt(vel_norm_scaler.var_.item())
        dns = dns / np.sqrt(prs_vel_norm_scaler.var_.item())
        #dns = dns / np.sqrt(pre_scaler.var_.item())
        #dns = dns.to(device)
        
        
        # normalize div
        div = div.view(-1, 1)
        #div_scaler.partial_fit(div.cpu().numpy())
        div = (div - prs_div_scaler.mean_.item()) / np.sqrt(prs_div_scaler.var_.item())
        
        pre = pre.view(-1, 1)
        #pre_scaler.partial_fit(pre.cpu().numpy())
        pre = (pre - prs_pre_scaler.mean_.item()) / np.sqrt(prs_pre_scaler.var_.item())
        

        fs = fs.view(-1, 1)
        #fs = fs / np.sqrt(vel_norm_scaler.var_.item())

          #pre = pre.to(device)
        #vel= (1-fs)*vel
        #feat = torch.cat((pre, vel, div, fs), dim=1)
        feat = torch.cat((vel, div, pre), dim=1)
        #print ("ING21")

        pred = model.forward(feat, pos)
        #model = None
        #print ("ING22")
        pres = pred.cpu().numpy() * np.sqrt(prs_scaler.var_.item()) + prs_scaler.mean_.item()


    torch.cuda.empty_cache()
    
    return pres
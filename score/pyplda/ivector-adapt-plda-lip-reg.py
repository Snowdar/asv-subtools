# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2020-05-31)

import numpy as np
import os
import sys

sys.path.insert(0, 'subtools/pytorch')

import libs.support.kaldi_io as kaldi_io
from plda_base import PLDA


class LIPReg(object):
    """
    Reference:
    Wang Q, Okabe K, Lee K A, et al. A Generalized Framework for Domain Adaptation of PLDA in Speaker Recognition[C]//ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020: 6619-6623.
    """
    def __init__(self, 
                 interpolation_weight=0.6):

        self.interpolation_weight = interpolation_weight

    def interpolation(self,plda_out_domain,plda_in_domain):
        
        _,between_var_out,within_var_out = self.plda_read(plda_out_domain)
        mean_in,between_var_in,within_var_in = self.plda_read(plda_in_domain)
        dim = mean_in.shape[0]

        S_w = within_var_out
        S_b = between_var_out

        eigh_w,Q_w = np.linalg.eigh(within_var_in)
        self.sort_svd(eigh_w, Q_w)
        eigh_diag_w = np.linalg.inv(np.diag(np.sqrt(eigh_w)))
        transform_com_w = np.matmul(eigh_diag_w,Q_w.T)
        E_w,P_w = np.linalg.eigh(np.matmul(np.matmul(transform_com_w,S_w),transform_com_w.T))
        B_w =np.matmul(np.matmul(Q_w,eigh_diag_w),P_w)
        self.within_var = within_var_in + (1-self.interpolation_weight)* np.matmul(np.matmul(np.linalg.inv(B_w).T,np.maximum(0,np.diag(E_w)-np.eye(dim))),np.linalg.inv(B_w))

        eigh_b,Q_b = np.linalg.eigh(between_var_in)
        self.sort_svd(eigh_b, Q_b)
        eigh_diag_b = np.linalg.inv(np.diag(np.sqrt(eigh_b)))
        transform_com_b = np.matmul(eigh_diag_b,Q_b.T)
        E_b,P_b = np.linalg.eigh(np.matmul(np.matmul(transform_com_b,S_b),transform_com_b.T))
        B_b =np.matmul(np.matmul(Q_b,eigh_diag_b),P_b)
        self.between_var = between_var_in + (1-self.interpolation_weight)* np.matmul(np.matmul(np.linalg.inv(B_b).T,np.maximum(0,np.diag(E_b)-np.eye(dim))),np.linalg.inv(B_b))
        self.mean = mean_in

    def plda_read(self,plda):
      
        with kaldi_io.open_or_fd(plda,'rb') as f:
            for key,vec in kaldi_io.read_vec_flt_ark(f):
                if key == 'mean':
                    mean = vec.reshape(-1,1)
                    dim = mean.shape[0]
                elif key == 'within_var':
                    within_var = vec.reshape(dim, dim)
                else:
                    between_var = vec.reshape(dim, dim)

        return mean,between_var,within_var

    def sort_svd(self,s, d):
      
        for i in range(len(s)-1):
            for j in range(i+1,len(s)):
                if s[i] > s[j]:
                    s[i], s[j] = s[j], s[i]
                    d[i], d[j] = d[j], d[i]

def main():

    if len(sys.argv)!=4:
        print('<plda-out-domain> <plda-in-domain> <plda-adapt> \n',
            )  
        sys.exit() 

    plda_out_domain = sys.argv[1]
    plda_in_domain = sys.argv[2]
    plda_adapt = sys.argv[3]

    lipreg=LIPReg()
    lipreg.interpolation(plda_out_domain,plda_in_domain)

    plda_new = PLDA()
    plda_new.mean = lipreg.mean
    plda_new.within_var = lipreg.within_var
    plda_new.between_var = lipreg.between_var
    plda_new.get_output()
    plda_new.plda_trans_write(plda_adapt)

if __name__ == "__main__":
    main()
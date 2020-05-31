# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2020-05-31)

import numpy as np
import os
import sys

sys.path.insert(0, 'subtools/pytorch')

import libs.support.kaldi_io as kaldi_io
from plda_base import PLDA


class LIP(object):
    """
    Reference:
    D.Garcia-RomeroandA.McCree, “Superviseddomain adaptation for i-vector based speaker recognition,” in Proc. IEEE ICASSP, 2014.
    """
    def __init__(self, 
                 interpolation_weight=0.4):

        self.interpolation_weight = interpolation_weight

        
    def interpolation(self,plda_out_domain,plda_in_domain):
        

        _,between_var_out,within_var_out = self.plda_read(plda_out_domain)
        mean_in,between_var_in,within_var_in = self.plda_read(plda_in_domain)

        self.mean = mean_in
        self.between_var = self.interpolation_weight*between_var_out+(1-self.interpolation_weight)*between_var_in
        self.within_var = self.interpolation_weight*within_var_out+(1-self.interpolation_weight)*within_var_in

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

def main():

    if len(sys.argv)!=4:
        print('<plda-out-domain> <plda-in-domain> <plda-adapt> \n',
            )  
        sys.exit() 

    plda_out_domain = sys.argv[1]
    plda_in_domain = sys.argv[2]
    plda_adapt = sys.argv[3]

    lip=LIP()
    lip.interpolation(plda_out_domain,plda_in_domain)

    plda_new = PLDA()
    plda_new.mean = lip.mean
    plda_new.within_var = lip.within_var
    plda_new.between_var = lip.between_var
    plda_new.get_output()
    plda_new.plda_trans_write(plda_adapt)

if __name__ == "__main__":
    main()
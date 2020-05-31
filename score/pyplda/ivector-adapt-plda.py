import numpy as np
import os
import sys
sys.path.insert(0, 'subtools/pytorch')
import libs.support.kaldi_io as kaldi_io
from plda import PLDA,PldaUnsupervisedAdaptor

# author JFZhou 2020-05-31

'''
Reference:https://github.com/kaldi-asr/kaldi/blob/master/src/ivectorbin/ivector-adapt-plda.cc
'''

def main():

    if len(sys.argv)!=4:
        print('<plda> <adapt-ivector-rspecifier> <plda-adapt> \n',
            )  
        sys.exit() 

    plda = sys.argv[1]
    train_vecs_adapt = sys.argv[2]
    plda_adapt = sys.argv[3]

    plda_new = PLDA()
    plda_new.plda_read(plda)
    plda_new.get_output()

    aplda_model=PldaUnsupervisedAdaptor()
    for _,vec in kaldi_io.read_vec(train_vecs_adapt):
        aplda_model.add_stats(1,vec)
    aplda_model.update_plda(plda_new)
    plda_new.plda_trans_write(plda_adapt)

if __name__ == "__main__":
    main()
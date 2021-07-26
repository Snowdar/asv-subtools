#coding:utf-8
import numpy as np
import sys
import os
sys.path.insert(0, 'subtools/pytorch')
import libs.support.kaldi_io as kaldi_io



# Copyright xmuspeech （Author:JFZhou,Fuchuan Tong 2021-1-5)



'''
s = φ^T_1Λφ_2 + φ^T_2 Λφ_1 + φ^T_1Γφ_1 + φ^T_2Γφ_2 +(φ_1 + φ_2)T_c + k
where
Γ = −1/4(Σ_{wc} + 2Σ_{ac})^{−1} − 1/4Σ^{−1}_{wc} + 1/2Σ^{−1}_{tot}
Λ = −1/4(Σ_{wc} +2Σ_{ac})^{−1} + 1/4Σ^{−1}_{wc}
c = (( Σ_{wc} +2Σ_{ac})^{−1} −Σ^{−1}_{tot})μ
k = log|Σtot|−1/2log|Σ_{wc} +2Σ_{ac}|−1/2log|Σ_{wc}|+μ^T(Σ^{−1}_{tot} −(Σ_{wc} +2Σ_{ac})^{−1})μ. 
'''

def PLDAScoring(enroll_ivector,test_ivector,Gamma,Lambda,c,k):

    score = np.matmul(np.matmul(enroll_ivector.T,Lambda),test_ivector) + np.matmul(np.matmul(test_ivector.T,Lambda),enroll_ivector) \
        + np.matmul(np.matmul(enroll_ivector.T,Gamma),enroll_ivector) + np.matmul(np.matmul(test_ivector.T,Gamma),test_ivector) \
        + np.matmul((enroll_ivector + test_ivector).T,c) + k
    
    return score[0][0]

def CalculateVar(between_var,within_var,mean):

    total_var_inv = np.linalg.inv(between_var + within_var)
    wc_add_2ac_inv = np.linalg.inv(within_var + 2* between_var)
    wc_inv = np.linalg.inv(within_var)

    # Gamma
    Gamma = (-1/4)*(wc_add_2ac_inv+wc_inv)+(1/2)*total_var_inv

    #Lambda
    Lambda = (-1/4)*(wc_add_2ac_inv-wc_inv)

    #c
    c = np.matmul((wc_add_2ac_inv-total_var_inv),mean)
    
    # Since k is a constant for the addition of all scores and does not affect the eer value, it is not counted as a scoring term
    # k = logdet_tot -(1/2) *(logdet_w_two_b+logdet_w) + np.matmul(np.matmul(mean.T,np.linalg.inv(total_var)-np.linalg.inv(within_var + 2* between_var)),mean)
    k = 0

    return Gamma,Lambda,c,k


def main(plda,train_ivector,test_ivector,trials,scores):

    with kaldi_io.open_or_fd(plda,'rb') as f:
        for key,vec in kaldi_io.read_vec_flt_ark(f):
            if key == 'mean':
                mean = vec.reshape(-1,1)
                dim = vec.shape[0]
            elif key == 'within_var':
                within_var = vec.reshape(dim, dim)
            else:
                between_var = vec.reshape(dim, dim)

    within_var = within_var + 5e-5*np.eye(within_var.shape[0])

    Gamma,Lambda,c,k = CalculateVar(between_var,within_var,mean)

    f_writer = open(scores,'w')
    enrollutt2vector = {}
    for key,vector in kaldi_io.read_vec(train_ivector):
        enrollutt2vector[key] = vector

    testutt2vector = {}
    for key,vector in kaldi_io.read_vec(test_ivector):
        testutt2vector[key] = vector 

    with open(trials,'r') as f:
        for line in f:
            enroll,test,_= line.strip().split()

            score = PLDAScoring(enrollutt2vector[enroll].reshape(-1,1),testutt2vector[test].reshape(-1,1),\
                                                                Gamma,Lambda,c,k)
            f_writer.write(enroll+' '+test+' '+str(score) +'\n')
    f_writer.close()


if __name__ == "__main__":
  
    if len(sys.argv) != 6:
        print("Usage: "+sys.argv[0]+" <plda> <train-ivector-rspecifier> <test-ivector-rspecifier> <trials-rxfilename> <scores-wxfilename>")
        print("e.g.: "+sys.argv[0]+" --num-utts=ark:exp/train/num_utts.ark plda ark:exp/train/spk_ivectors.ark ark:exp/test/ivectors.ark trials scores")
        exit(1)

    plda = sys.argv[1]
    train_ivector = sys.argv[2]
    test_ivector = sys.argv[3]
    trials = sys.argv[4]
    scores = sys.argv[5]

    main(plda,train_ivector,test_ivector,trials,scores)
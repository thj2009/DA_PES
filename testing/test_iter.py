# -*- coding: utf-8 -*-


import numpy as np
np.set_printoptions(threshold='nan')
import math
from scipy import sparse
import time

class Node(object):
    def __init__(self, data):
        self.data = data 
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
        
def combination_print(tree, XX, ll, M):
#    print tree.data
    ll_copy = ll[:]         # store the information of parents
    if tree.children != []:
        if tree.data != None:       
            ll_copy.append(tree.data)
        for child in tree.children:
            combination_print(child, XX, ll_copy, M)    
    else:
        if tree.data != None:
            ll_copy.append(tree.data)
    #        if len(ll_copy) == M:
            XX.append(ll_copy)
        
    return(XX, ll_copy)
        
def Cert_Combination(tree, M, pt, jt):
    # Construct a tree with dimension M, total order pt, interaction jt
    if M == 1:
        nn = Node(pt)
        tree.add_child(nn)
    else:
        for ii in range(pt+1):
            next_pt = pt - ii
            if ii == 1:
                next_jt = jt - 1
            else:
                next_jt = jt
                
              
            if (next_jt < M - 1 or (next_jt==M-1 and next_pt == next_jt)) \
               and next_jt <= next_pt and \
            next_pt - next_jt != 1 and\
            next_jt >= 0 :
                nn = Node(ii)
                tree.add_child(nn)
                Cert_Combination(nn, M-1, next_pt, next_jt)     
                
def Cert_Combi_cal(M, order, jt, max_order, qnorm):
    # Return the Combination
    # dimension M: total order: order
    # interaction: jt
    # with qnorm less than max_order
    XX = [] 
    if M >= jt:
        test = np.zeros(M)
        for ii in range(jt):
            test[ii] = 1
        if M != jt:
            test[jt] = order - jt
        min_norm = np.linalg.norm(test, qnorm)
          
        if min_norm < max_order + 1e-7:
       
            ll = []
            tree = Node(None)
            Cert_Combination(tree, M, order, jt)
                
            XX, ll = combination_print(tree, XX, ll, M)    
            AA = np.array(XX)
            AAnorm = [np.linalg.norm(AA[ii, :], qnorm) for ii in range(len(XX))]
            AAnorm = np.array(AAnorm)
            sat_ = np.argwhere(AAnorm <= max_order).T[0]
            XX = [XX[ii] for ii in sat_]
    return XX

def Tot_Combi_cal(M, order, qnorm):
    # return the Combination
    # dimension M: total order: order
    # with qnorm less thn order
    XX = []
    for ii in range(order + 1):
        for jt in range(ii+1):
#            print Cert_Combi_cal(M, ii, jt, order, qnorm)
            XX += Cert_Combi_cal(M, ii, jt, order, qnorm)
    return XX            

def Indi_Combi_cal(M, order, Max_order, qnorm):
    XX = []
    for jt in range(order+1):
#            print Cert_Combi_cal(M, ii, jt, order, qnorm)
        XX += Cert_Combi_cal(M, order, jt, Max_order, qnorm)
    return XX            
   
    
    
if __name__ == '__main__':
    M = 21
    order = 3
    jt = 2
#    max_order = 8
    qnorm = 1
    
#    tic = time.clock()
#    XX = []
#    XX = Tot_Combi_cal(M, order, qnorm)   
#    print XX
#    print len(XX)
#    print math.factorial(M+order)/(math.factorial(order)*math.factorial(M))
#    toc = time.clock()
#    print '==================='
#    print 'Time = %f' %((toc-tic)/60.)
    for i in range(order + 1):
        print(len(Indi_Combi_cal(M, i, order, qnorm)))
#    print(Indi_Combi_cal(M, i, order, qnorm))
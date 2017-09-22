# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:54:30 2017

@author: rflamary
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


import numpy as np
import sklearn
import scipy.optimize as spo
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel

import time
__time_tic_toc=time.time()


def get_label_matrix(y):
    vals=np.unique(y)
   
    # class matrices for source
    Y=np.zeros((len(y),len(vals)))
    Yb=np.zeros((len(y),len(vals)))
    for i,val in enumerate(vals):
        Y[:,i]=2*((y==val)-.5)
        Yb[:,i]=(y==val)
    return Y,Yb

def estimGamma(X):
    return 1./(2*(np.median(cdist(X,X,'euclidean'))**2))


def tic():
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    t=time.time()
    return t-__time_tic_toc

def loss_hinge(Y,F):
    res=np.zeros((Y.shape[0],F.shape[0]))
    for i in range(Y.shape[1]):
        res+=np.maximum(0,1-Y[:,i].reshape((Y.shape[0],1))*F[:,i].reshape((1,F.shape[0])))**2
    return res

class Classifier:    
    # cross validate parameters with k-fold classification
   
    
    def crossval(self,X,Y,kerneltype='linear',nbsplits=5,g_range=np.logspace(-3,3,7),l_range = np.logspace(-3,0,4)):
        kf = KFold(n_splits=nbsplits)
        if kerneltype=='rbf':
            dim=(len(g_range),len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)
    
            for i,g in enumerate(g_range):
                for j,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.rbf_kernel(X[train,:],gamma=g)
                        Kt=sklearn.metrics.pairwise.rbf_kernel(X[train,:],X[test,:],gamma=g)
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)
                        
                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)

                        results[i,j] += np.mean(ydec==yt)
            results = results /nbsplits
            #print results
    
            i,j = np.unravel_index(results.argmax(), dim)
            
            self.lambd=l_range[j]

            return g_range[i],l_range[j]
        else:
            dim=(len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)
            for i,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.linear_kernel(X[train,:])
                        Kt=sklearn.metrics.pairwise.linear_kernel(X[train,:],X[test,:])
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)
                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)
                        results[i] += np.mean(ydec==yt)
            results = results /nbsplits

    
            self.lambd=l_range[results.argmax()]

            return self.lambd


def hinge_squared_reg(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[0],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()
  
def hinge_squared_reg_bias(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[1],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    w[:,-1]=0
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()
    


class SVMClassifier(Classifier):

    def __init__(self,lambd=1e-2,bias=False):
        self.lambd=lambd
        self.w=None
        self.bias=bias


    def fit(self,K,y):  
        # beware Y is a binary matrix to allow for more general solvers (see JDOT)
        if self.bias:
            K1=np.hstack((K,np.ones((K.shape[0],1))))
            self.w=np.zeros((K1.shape[1],y.shape[1]))
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg_bias(w,X=K1,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)            
            self.b=self.w.reshape((K1.shape[1],y.shape[1]))[-1,:]
            self.w=self.w.reshape((K1.shape[1],y.shape[1]))[:-1,:]

        else:
            self.w=np.zeros((K.shape[1],y.shape[1]))
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg(w,X=K,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)            
            self.w=self.w.reshape((K.shape[1],y.shape[1]))

    def predict(self,K):
        if self.bias:
            return np.dot(K,self.w)+self.b
        else:
            return np.dot(K,self.w)



class KRRClassifier(Classifier):

    def __init__(self,lambd=1e-2):
        self.lambd=lambd

    def fit(self,K,y,sw=False):
        ns=K.shape[0]
        if sw:
            K=K*sw
        K0=np.vstack((np.hstack((np.eye(ns),np.zeros((ns,1)))),np.zeros((1,ns+1))))
        
        ## true reg in RKHS
        #K0=np.vstack((np.hstack((K,np.zeros((ns,1)))),np.zeros((1,ns+1))))

        K1=np.hstack((K,np.ones((ns,1))))
        if sw:
            y1=K1.T.dot(y*sw)
        else:
            y1=K1.T.dot(y)

        temp=np.linalg.solve(K1.T.dot(K1) + self.lambd*K0,y1)
        self.w,self.b=temp[:-1],temp[-1]

    def predict(self,K):
        return np.dot(K,self.w)+self.b

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:59:10 2017

@author: rflamary
"""

import numpy as np
from scipy.spatial.distance import cdist 

import classif
import sklearn

import ot

#from sklearn import datasets


# X: source domain
# y: source labeks
# Xtest: target domain
# ytest is optionnal, just to measure performances of the method along iterations
# gamma: RBF kernel param (default=1)
# numIterBCD: number of Iterations for BCD (default=10)
# alpha: ponderation between ground cost + function cost
# method: choice of algorithm for transport computation (default: LP)


class TlossKRR_DA:
    def __init__(self,kerneltype='rbf',alpha=1,numIterBCD=10, gamma=1, lambd=1e1, method='emd', reg=1):
        self.ktype=kerneltype
        self.gamma=gamma
        self.lambd=lambd
        self.alpha=alpha
        self.method=method
        self.numIterBCD=numIterBCD
        self.reg=reg
        
    def fit(self,X,y,Xt, yt=[],lambd_f=None,gamma_f=None):
        self.Xt=Xt
        
        if self.ktype=='rbf':
            if not lambd_f:
                gamma_f,lambd_f = classif.crossval(X,y,kerneltype='rbf')
            K=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_f)
        else:
            if not lambd_f:
                lambd_f = classif.crossval(X,y,kerneltype='linear')
            K=sklearn.metrics.pairwise.linear_kernel(X)
        
        f = classif.KRRClassifier(lambd_f)
        f.fit(K,y)
        
        self.tloss=computeTLOT_KRR(X,y,Xt, f=f, ytest=yt,
                                   gamma_f=gamma_f, 
                                   gamma_g=self.gamma, 
                                   numIterBCD = self.numIterBCD, 
                                   alpha=self.alpha, 
                                   lambd=self.lambd, 
                                   method=self.method,
                                   reg=self.reg,
                                   ktype=self.ktype)
        
    def predict(self,Xtest,full_output=False):
        if self.ktype=='rbf':
            Ktest = sklearn.metrics.pairwise.rbf_kernel(self.Xt,Xtest,gamma=self.gamma)
        else:
            Ktest = sklearn.metrics.pairwise.linear_kernel(self.Xt,Xtest)
            
        ypred=self.tloss['clf'].predict(Ktest.T)
        ydec=np.argmax(ypred,1)+1   
        if full_output: 
            return ydec,self.tloss
        else:
            return ydec


def computeTLOT_KRR_no_f(X,y,Xtest,gamma_g=1, numIterBCD = 10, alpha=1,lambd=1e1, method='emd',reg=1,ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #print np.max(C0)
    C0=C0/np.median(C0)

    # classifier    
    g = classif.KRRClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,Xtest,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest,Xtest)

    C = alpha*C0#+ cdist(y,ypred,metric='sqeuclidean')
    k=0
    while (k<numIterBCD):# and not changeLabels:
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)

        Yst=ntest*G.T.dot(y)

        g.fit(Kt,Yst)
        ypred=g.predict(Kt)
       
        # function cost
        fcost = cdist(y,ypred,metric='sqeuclidean')

        C=alpha*C0+fcost
            
    return g, np.sum(G*(fcost))    
    
def computeTLOT_KRR(X,y,Xtest,f, 
                    ytest=[], gamma_f=1,gamma_g=1, numIterBCD = 10, alpha=1,
                    lambd=1e1, method='emd',reg=1,ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #C0=C0/np.median(C0)
    C0=C0/np.max(C0)

    # classifier    
    g = classif.KRRClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        K=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_f)
        K2=sklearn.metrics.pairwise.rbf_kernel(Xtest,X,gamma=gamma_f)
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,Xtest,gamma=gamma_g)
    else:
        K=sklearn.metrics.pairwise.linear_kernel(X)
        K2=sklearn.metrics.pairwise.linear_kernel(Xtest,X)
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest,Xtest)
        
    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    #Init initial g(.)

    ypred=f.predict(K2)
    y_init=f.predict(K)

    C = alpha*C0+ cdist(y_init,ypred,metric='sqeuclidean')

    # do it only if the final labels were given
    if len(ytest):
        ydec=np.argmax(ypred,1)+1
        TBR1=np.mean(ytest==ydec)
        TBR.append(TBR1)

    k=0
    changeLabels=False
    while (k<numIterBCD):# and not changeLabels:
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)

        Yst=ntest*G.T.dot(y)

        g.fit(Kt,Yst)
        ypred=g.predict(Kt)
       
        # function cost
        fcost = cdist(y_init,ypred,metric='sqeuclidean')

        C=alpha*C0+fcost

        ydec_tmp=np.argmax(ypred,1)+1
        #if k>1:
            #changeLabels=np.all(ydec_tmp==ydec)
        ydec=ydec_tmp
        if len(ytest):
            TBR1=np.mean(ytest==ydec)
            TBR.append(TBR1)
            
        sav_fcost.append(np.sum(G*fcost))
        sav_totalcost.append(np.sum(G*(alpha*C0+fcost)))

    results['ypred0']=ypred
    results['ypred']=np.argmax(ypred,1)+1
    if len(ytest):
        results['TBR']=TBR
    results['f']=f
    results['clf']=g
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost
    return results

def jdot_svm(X,y,Xtest,  
                      ytest=[],gamma_g=1, numIterBCD = 10, alpha=1,
                      lambd=1e1, method='emd',reg_sink=1,ktype='linear'):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    #C0=C0/np.max(C0)
    #C0=C0/np.median(C0)

    # classifier    
    g = classif.SVMClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(Xtest,gamma=gamma_g)
        #Ks=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(Xtest)
        #Ks=sklearn.metrics.pairwise.linear_kernel(X)
        
    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    
    ypred=np.zeros(y.shape)
#    ypred=-np.ones(y.shape)
#    ypred[:,0]=1
    #ypred=np.random.rand(y.shape[0],y.shape[1])*2-1
    #Chinge=SVMclassifier.loss_hinge(y,ypred)
    
    Chinge=np.zeros(C0.shape)
    C=alpha*C0+Chinge
    
    # do it only if the final labels were given
    if len(ytest):
        TBR.append(np.mean(ytest==np.argmax(ypred,1)+1))

    k=0
    while (k<numIterBCD):
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg_sink)
        if method=='emd':
            G=  ot.emd(wa,wb,C)

        if k>1:
            sav_fcost.append(np.sum(G*Chinge))
            sav_totalcost.append(np.sum(G*(alpha*C0+Chinge)))

            
        Yst=ntest*G.T.dot((y+1)/2.)
        #Yst=ntest*G.T.dot(y_f)
        g.fit(Kt,Yst)
        ypred=g.predict(Kt)

        
        Chinge=classif.loss_hinge(y,ypred)
        #Chinge=SVMclassifier.loss_hinge(y_f*2-1,ypred*2-1)
        
        C=alpha*C0+Chinge

        if len(ytest):
            TBR1=np.mean(ytest==np.argmax(ypred,1)+1)
            TBR.append(TBR1)
            

    results['ypred']=np.argmax(ypred,1)+1
    if len(ytest):
        results['TBR']=TBR

    results['clf']=g
    results['G']=G
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost
    return results

def computeTLOT_NN(X,Y,Xtest, ytest=[], numIterBCD = 10, alpha=1, num_hidden=50,method='emd',reg=1,nb_epoch=100,batch_size=10):
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    C0=C0/np.max(C0)

    # classifier    
    g = classif.NNClassifier(X.shape[1],num_hidden,Y.shape[1])
        
    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    #Init initial g(.)
    g.fit(X,Y,nb_epoch=nb_epoch,batch_size=batch_size)
    ypred=g.predict(Xtest)

    C = alpha*C0+ cdist(Y,ypred,metric='sqeuclidean')

    # do it only if the final labels were given
    if len(ytest):
        ydec=np.argmax(ypred,1)+1
        TBR1=np.mean(ytest==ydec)
        TBR.append(TBR1)

    k=0
    changeLabels=False
    while (k<numIterBCD):# and not changeLabels:
        k=k+1
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        if method=='emd':
            G=  ot.emd(wa,wb,C)

        Yst=ntest*G.T.dot(Y)
            
        g = classif.NNClassifier(X.shape[1],num_hidden,Y.shape[1])

        g.fit(Xtest,Yst,nb_epoch=nb_epoch,batch_size=batch_size)
        ypred=g.predict(Xtest)
        
        # function cost
        fcost = cdist(Y,ypred,metric='sqeuclidean')
        #pl.figure()
        #pl.imshow(fcost)
        #pl.show()

        C=alpha*C0+fcost

        ydec_tmp=np.argmax(ypred,1)+1
        if k>1:
            changeLabels=np.all(ydec_tmp==ydec)
            sav_fcost.append(np.sum(G*fcost))
            sav_totalcost.append(np.sum(G*(alpha*C0+fcost)))

        ydec=ydec_tmp
        if len(ytest):
            TBR1=np.mean(ytest==ydec)
            TBR.append(TBR1)
            
    results['ypred0']=ypred
    results['ypred']=np.argmax(ypred,1)+1
    if len(ytest):
        results['TBR']=TBR
    results['clf']=g
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost
    return results    
    
    
    
def computeTLOTjoint(X,y,X2, y2, gamma=1,eps=1, numIterBCD = 10, alpha=1, lambd=1e1, method='emd',reg=1):
    # Initializations
    n = X.shape[0]
    n2 = X2.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((n2,))/n2

    # original loss
    C0=cdist(X,X2,metric='sqeuclidean')

    C0=C0/np.max(C0)#+np.random.rand(C0.shape[0],C0.shape[1])
    #C=C0
    #C0=np.random.rand(C0.shape[0],C0.shape[1])

    # classifier
    f1 = classif.KRRClassifier(lambd)
    f2 = classif.KRRClassifier(lambd)

    # compute kernels
    K=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma)
    K2=sklearn.metrics.pairwise.rbf_kernel(X2,X2,gamma=gamma)


    TBR = []

    results = {}

    #Init initial g(.)

    f1.fit(K,y)
    f2.fit(K2,y2)

    ypred1=f1.predict(K)
    ypred2=f2.predict(K2)


    C = alpha*C0+ cdist(ypred1,ypred2,metric='sqeuclidean')

    for k in range(numIterBCD):
        # update G
        if method=='sinkhorn':
            G = ot.sinkhorn(wa,wb,C,reg)
        elif method=='emd':
            G=ot.emd(wa,wb,C)

        # update f1
        Y1est=n*G.dot(ypred2)
        yobj=(y+eps*Y1est)/(1+eps)# awesome l2 los!!!
        f1.fit(K,yobj)
        ypred1=f1.predict(K)

        # update f2
        Y2est=n2*G.T.dot(ypred1)
        yobj2=(y2+eps*Y2est)/(1+eps) # awesome l2 los!!!
        f2.fit(K2,yobj2)
        ypred2=f2.predict(K2)

        # function cost
        fcost = cdist(ypred1,ypred2,metric='sqeuclidean')
#        pl.figure()
#        pl.imshow(fcost,interpolation='nearest')
#        pl.colorbar()
#        pl.show()

        C=alpha*C0+fcost


    results['ypred1']=ypred1
    results['ypred2']=ypred2

    results['f1']=f1
    results['f2']=f2


    return results
    

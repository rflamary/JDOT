# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:59:10 2017

@author: rflamary
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

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
# method: choice of algorithm for transport computation (default: emd)


def jdot_krr(X,y,Xtest,gamma_g=1, numIterBCD = 10, alpha=1,lambd=1e1, 
             method='emd',reg=1,ktype='linear'):
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
            
    return g,np.sum(G*(fcost))    
    

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
    return g,results
#
def jdot_nn_l2(get_model,X,Y,Xtest,ytest=[],fit_params={},reset_model=True, numIterBCD = 10, alpha=1,method='emd',reg=1,nb_epoch=100,batch_size=10):
    # get model should return a new model compiled with l2 loss
    
    
    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa=np.ones((n,))/n
    wb=np.ones((ntest,))/ntest

    # original loss
    C0=cdist(X,Xtest,metric='sqeuclidean')
    C0=C0/np.max(C0)

    # classifier    
    g = get_model()
        
    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    #Init initial g(.)
    g.fit(X,Y,**fit_params)
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
        
        if reset_model:
            g=get_model()

        g.fit(Xtest,Yst,**fit_params)
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
            TBR1=np.mean((ytest-ypred)**2)
            TBR.append(TBR1)
            
    results['ypred0']=ypred
    results['ypred']=np.argmax(ypred,1)+1
    if len(ytest):
        results['mse']=TBR
    results['clf']=g
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost
    return g,results    
    


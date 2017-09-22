# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 16:52:46 2014

@author: flam
"""


import numpy as np
import pylab as pl

import jdot

#from sklearn import datasets
import sklearn
from scipy.spatial.distance import cdist 
import ot


#%% data generation

seed=1985
np.random.seed(seed)

n = 200
ntest=200
nz=.3

theta=0.8


n2=int(n/2)
sigma=0.05

xs=np.random.randn(n,1)+2
xs[:n2,:]-=4
ys=sigma*np.random.randn(n,1)+np.sin(xs/2)

fs_s = lambda x: np.sin(x/2)

xt=np.random.randn(n,1)+2
xt[:n2,:]/=2 
xt[:n2,:]-=3
  
gauss = lambda x,s,m: np.exp((x-m)**2/(2*s*s))/(s*np.sqrt(2*np.pi))

mus_x = lambda x: gauss(x,1,2)/2+gauss(x,1,-2)/2


yt=sigma*np.random.randn(n,1)+np.sin(xt/2)
xt+=2

fs_t = lambda x: np.sin((x-2)/2)

mut_x = lambda x: gauss(x,1,2)/2+gauss(x,1./2,-4)/2
                       
xvisu=np.linspace(-4,6.5,100)

pl.figure(1)
pl.clf()

pl.subplot()
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend()
pl.title('Toy regression example')
#pl.savefig('imgs/visu_data_reg.eps')

#%% TLOT
lambd0=1e1
itermax=15
gamma=1e-1
alpha=1e0/4
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,yt,metric='sqeuclidean')
C=alpha*C0+fcost
G0=ot.emd(ot.unif(n),ot.unif(n),C)

model,loss = jdot.jdot_krr(xs,ys,xt,gamma_g=gamma,numIterBCD = 10, alpha=alpha, lambd=lambd0,ktype='rbf')

K=sklearn.metrics.pairwise.rbf_kernel(xt,xt,gamma=gamma)
Kvisu=sklearn.metrics.pairwise.rbf_kernel(xvisu.reshape((-1,1)),xt,gamma=gamma)
ypred=model.predict(Kvisu)
ypred0=model.predict(K)


# compute true OT
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,ypred0,metric='sqeuclidean')
C=alpha*C0+fcost
G=ot.emd(ot.unif(n),ot.unif(n),C)


pl.figure(2)
pl.clf()
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
pl.plot(xvisu,ypred,'r',label='Tloss model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend()
pl.title('Toy regression example')


#%%
seed=1985
np.random.seed(seed)
idv=np.random.permutation(n)
fs=17
nb=15
pl.figure(3,(10,7))
pl.clf()

pl.subplot(2,2,1)
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),label='Source model')
pl.plot(xvisu,fs_t(xvisu),label='Target model')
#pl.xlabel('x')

pl.ylabel('y')
pl.legend(loc=4,fontsize=.7*fs)
pl.title('Toy regression example',fontsize=fs)

pl.subplot(2,2,2)
pl.scatter(xs,ys,edgecolors='k')
pl.scatter(xt,yt,edgecolors='k')
#pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
#pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
for i in range(nb):
    idt=G0[idv[i],:].argmax()
    if not i:
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k',label='OT matrix link')
    else:
        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k')

#pl.xlabel('x')

pl.ylabel('y')
pl.legend(loc=4,fontsize=.7*fs)
pl.title('OT matrix on joint distribution',fontsize=fs)

pl.subplot(2,2,3)
pl.scatter(xs,ys,edgecolors='k')
pl.scatter(xt,yt,edgecolors='k')
#pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
#pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
for i in range(nb):
    idt=G[idv[i],:].argmax()
    
    pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k')

pl.xlabel('x')

pl.ylabel('y')
pl.legend(loc=4)
pl.title('OT matrix of JDOT',fontsize=fs)

pl.subplot(2,2,4)
pl.scatter(xs,ys,edgecolors='k')
pl.scatter(xt,yt,edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),label='Source model')
pl.plot(xvisu,fs_t(xvisu),label='Target model')
pl.plot(xvisu,ypred,'g',label='JDOT model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend(loc=4,fontsize=.7*fs)
pl.title('Model estimated with JDOT',fontsize=fs)

pl.tight_layout()
#pl.savefig('imgs/visu_reg.eps')


#%%
#seed=1985
#np.random.seed(seed)
#idv=np.random.permutation(n)
#fs=12
#nb=15
#pl.figure(3,(12,3))
#pl.clf()
#
#pl.subplot(1,4,1)
#pl.scatter(xs,ys,label='Source samples',edgecolors='k')
#pl.scatter(xt,yt,label='Target samples',edgecolors='k')
#pl.plot(xvisu,fs_s(xvisu),label='Source model')
#pl.plot(xvisu,fs_t(xvisu),label='Target model')
##pl.xlabel('x')
#pl.xlabel('x')
#pl.ylabel('y')
#pl.legend(loc=4,fontsize=.7*fs)
#pl.title('Toy regression example',fontsize=fs)
#
#pl.subplot(1,4,2)
#pl.scatter(xs,ys,edgecolors='k')
#pl.scatter(xt,yt,edgecolors='k')
##pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
##pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
#for i in range(nb):
#    idt=G0[idv[i],:].argmax()
#    if not i:
#        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k',label='OT matrix link')
#    else:
#        pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k')
#
##pl.xlabel('x')
#
##pl.ylabel('y')
#pl.legend(loc=4,fontsize=.7*fs)
#pl.title('OT matrix on joint distribution',fontsize=fs)
#pl.xlabel('x')
#pl.subplot(1,4,3)
#pl.scatter(xs,ys,edgecolors='k')
#pl.scatter(xt,yt,edgecolors='k')
##pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
##pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
#for i in range(nb):
#    idt=G[idv[i],:].argmax()
#    
#    pl.plot([xs[idv[i]],xt[idt]],[ys[idv[i]],yt[idt]],'k')
#
#pl.xlabel('x')
#
##pl.ylabel('y')
#pl.legend(loc=4)
#pl.title('OT matrix of JDOT',fontsize=fs)
#
#pl.subplot(1,4,4)
#pl.scatter(xs,ys,edgecolors='k')
#pl.scatter(xt,yt,edgecolors='k')
#pl.plot(xvisu,fs_s(xvisu),label='Source model')
#pl.plot(xvisu,fs_t(xvisu),label='Target model')
#pl.plot(xvisu,ypred,'g',label='JDOT model')
#pl.xlabel('x')
#
##pl.ylabel('y')
#pl.legend(loc=4,fontsize=.7*fs)
#pl.title('Model estimated with JDOT',fontsize=fs)
#
#pl.tight_layout(pad=00,h_pad=0)
##pl.savefig('imgs/visu_reg2.eps')
##pl.savefig('imgs/visu_reg2.pdf')
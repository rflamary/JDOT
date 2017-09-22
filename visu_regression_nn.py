# -*- coding: utf-8 -*-
"""
Neural network regression example for JDOT
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


import numpy as np
import pylab as pl

import jdot

#from sklearn import datasets
from scipy.spatial.distance import cdist 
import ot
import keras


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


#%% learn on source

def get_model():
    # simple 1D nn
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(100,activation='tanh',input_shape=(1,)))
    model.add(keras.layers.Dense(50,activation='tanh'))
    model.add(keras.layers.Dense(1,activation='linear'))
    
    model.compile('sgd','mse')
    
    return model

model=get_model()

fit_params={'epochs':60}

model.fit(xs,ys,**fit_params)
ypred=model.predict(xvisu)

pl.figure(2)
pl.clf()
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
pl.plot(xvisu,ypred,'r',label='Source estimated model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend()
pl.title('Toy regression example')

#%% TLOT

itermax=5
alpha=1
C0=cdist(xs,xt,metric='sqeuclidean')
#print np.max(C0)
C0=C0/np.median(C0)
fcost = cdist(ys,yt,metric='sqeuclidean')
C=alpha*C0+fcost
G0=ot.emd(ot.unif(n),ot.unif(n),C)

fit_params={'epochs':100}

model,loss = jdot.jdot_nn_l2(get_model,xs,ys,xt,ytest=yt,fit_params=fit_params,numIterBCD = itermax, alpha=alpha)

ypred=model.predict(xvisu.reshape((-1,1)))


pl.figure(2)
pl.clf()
pl.scatter(xs,ys,label='Source samples',edgecolors='k')
pl.scatter(xt,yt,label='Target samples',edgecolors='k')
pl.plot(xvisu,fs_s(xvisu),'b',label='Source model')
pl.plot(xvisu,fs_t(xvisu),'g',label='Target model')
pl.plot(xvisu,ypred,'r',label='JDOT model')
pl.xlabel('x')

pl.ylabel('y')
pl.legend()
pl.title('Toy regression example')

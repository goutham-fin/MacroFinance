import sys
sys.path.insert(0, '../Models/')
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import matplotlib.ticker as mticker

from Benchmark.model_recursive_class import model_recursive
from Benchmark.model_class import model

import os

if __name__ == '__main__':
    params={'rhoE': 0.05, 'rhoH': 0.05, 'aE': 0.15, 'aH': 0.03,
            'alpha':0.75, 'kappa':5, 'delta':0.05, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':2, 'gammaH':2,'maxIterations':400,'nEpochs':2000}
    params['scale']=2
    params['active'] = 'on'
    model1 = model_recursive(params)
    #model1.maxIterations=15
    model1.solve()
    
    model1.f[650:]=model1.f[650]
    
    
    fig, ax1= plt.subplots()
    ax1.grid(False)
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax1.plot(model1.ssq[5:],color='b',label='Return volatility')
    ax1.set_xlabel('Wealth share')
    ax2.plot(model1.f,linestyle=':',color='g',label='Stationary distribution')
    fig.legend(loc='upper right',bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax2.set_yticks([])
    ax1.set_yticks([]) 
    plt.savefig('../output/plots/rp_dist.png')
    
    
    
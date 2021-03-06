# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:21:16 2020

@author: Goutham Gopalakrishna
"""

import sys
sys.path.insert(0,'../')
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from Simulation.simulation_benchmark import simulation_model
sys.path.insert(0,'../Models/')
import os
sys.path.insert(0,'../Models/')
from Benchmark.model_recursive_class import model_recursive
#plt.rcParams['axes.facecolor']='white'
#plt.rcParams['savefig.facecolor']='white' 
plt.rcParams['axes.grid']=False


if __name__=='__main__':
    params={'rhoE': 0.06, 'rhoH': 0.04, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':10, 'delta':0.02, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':1, 'gammaH':1,'load':False, 'scale':2,'utility':'recursive'}
    log = model_recursive(params)
    log.solve()
    
    params['gammaE'] = 5; params['gammaH'] = 5;  
    rec1 = model_recursive(params)
    rec1.solve()
    
    params['gammaE']= 20; params['gammaH'] = 20; 
    rec2 = model_recursive(params)
    rec2.solve()
    
    vars_list = ['q','psi','ssq','mu_z','sig_za','priceOfRiskE','rp','q','pd'] 
    obs = ['log','rec1','rec2']
    vars_ = []
    for v in vars_list:
        for o in obs:
            vars_.append(str(o) + '.' + str(v))
    labels = ['$q$','$\psi$','$\sigma_q + \sigma$','$\mu^z$','$\sigma^z$','$\zeta_{e}$', '$(\mu_e^R-r)$','$q$','$p-d$']
    title = ['Price','Capital Share: Experts', 'Price volatility','Drift of wealth share: Experts',\
                             'Volatility of wealth share: Experts', 'Market price of risk: Experts','Risk premium: Experts','Price','Log Price-Dividend ratio']
    try:
        if not os.path.exists('../output/plots'):
            os.mkdir('../output/plots') 
        plot_path = '../output/plots/'  
    except:
        print('Cannot make directory for plots')
    
      
    for i in range(len(vars_list)):
        last = 400
        plt.plot(eval(obs[0]).z[1:-last], eval(vars_[3*i])[1:-last,0],label='RA=1')
        plt.plot(eval(obs[1]).z[1:-last],eval(vars_[3*i+1])[1:-last,0],label='RA=5')
        plt.plot(eval(obs[2]).z[1:-last],eval(vars_[3*i+2])[1:-last,0],label='RA=20',color='b') 
        plt.grid(False)
        plt.legend(loc=0)
        plt.axis('tight')
        plt.xlabel('Wealth share (z)',fontsize=15)
        plt.ylabel(labels[i],fontsize=15)
        plt.title(title[i], fontsize = 20)
        plt.rc('legend', fontsize=15) 
        plt.rc('axes',labelsize = 15)
        #plt.rc('figure', titlesize=50)
        plt.savefig(plot_path + str(vars_list[i]) + '_benchmark.png')
        plt.figure()
    
    fig = plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    ax3=fig.add_subplot(111, label="3", frame_on=False)
    ax.plot(eval(obs[0]).z[10:479],pd.DataFrame(eval(obs[0]).f_norm).dropna()[10:-20], label='RA=1')
    ax2.plot(eval(obs[0]).z[10:479],pd.DataFrame(eval(obs[1]).f_norm).dropna()[10:-20], label='RA=5',color='g')
    ax3.plot(eval(obs[0]).z[10:479],pd.DataFrame(eval(obs[2]).f_norm).dropna()[10:-20], label='RA=20',color='b')
    ax.legend(loc=2,bbox_to_anchor=(0.8,0.9),fontsize=10)
    ax2.legend(loc=2,bbox_to_anchor=(0.8,0.8),fontsize=10)
    ax3.legend(loc=2,bbox_to_anchor=(0.8,0.7),fontsize=10)
    ax.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    plt.title('Stationary distribution',fontsize = 20)
    plt.xlabel('Wealth share (z)',fontsize=15)
    plt.savefig(plot_path + 'dist_benchmark.png')
    plt.show()
    
    
    
    #plot distribution of average crisis length
    home = os.path.expanduser('~')
    try:
        if not os.path.exists(os.path.join(home,'Desktop','pickles')):
            print('Cannot find pickles. Fun the file main.py first')
    except:
            pickle_path = os.path.join(home,'Desktop','pickles','')
    
    def read_pickle(filename):
        with open(str(filename) + '.pkl', 'rb') as f:
            return dill.load(f)
    
    sims = [0]*5
    for i in range(0,5):
        sims[i] = read_pickle(pickle_path+ 'sim' + str(i))
    
    sims_temp = np.arange(0,5)
    max_ = 0
    crisis_length = []
    for i in range(len(sims_temp)):
        #sims[sims_temp[i]].crisis_length_freq = np.array([(k, sum(1 for i in g)) for k,g in groupby(sims[sims_temp[i]].crisis_indicator_freq.transpose().reshape(-1))])
        #sims[sims_temp[i]].crisis_length_freq_mean = np.mean(sims[sims_temp[i]].crisis_length[~np.isnan(sims[sims_temp[i]].crisis_length_freq[:,0])][:,1])
        sims[sims_temp[i]].crisis_length_freq_hist = sims[sims_temp[i]].crisis_length_freq[~np.isnan(sims[sims_temp[i]].crisis_length_freq).any(axis=1),:][:,1]
        max_ = max(max_,sims[sims_temp[i]].crisis_length_freq_hist.shape[0])
    #create dataframe 
    crisis_length_freq_pd = pd.DataFrame(np.zeros([max_]))
    for i in range(len(sims_temp)):
        crisis_length_freq_pd=pd.concat([crisis_length_freq_pd,pd.Series(sims[sims_temp[i]].crisis_length_freq_hist)],axis=1)
    crisis_length_freq_pd = crisis_length_freq_pd.iloc[:,1:]
    crisis_length_freq_pd.columns = [x+1 for x in sims_temp]
    
    import joypy
    plt.figure(figsize=(16,5), dpi= 80)
    fig, axes = joypy.joyplot(pd.DataFrame(crisis_length_freq_pd),hist=True,bins=20,density=True)
    axes[-1].set_xlabel('Average length of crisis')
    
    #joypy.joyplot(pd.DataFrame(crisis_length_freq))
    
    plt.hist(sims[0].crisis_length,density=True), plt.hist(sims[1].crisis_length_hist,density=True),plt.hist(sims[2].crisis_length_hist,density=True)
    plt.xlim(0,15)
    
    labels_ = ['RA ='+str(sims_temp[i]+1) for i in range(5)]
    plt.hist(np.array(crisis_length_freq_pd),bins=200,label=labels_,density=True)
    plt.legend(loc='upper right')
    plt.xlim(1,20)
    plt.xlabel('Average length of crisis in Months',fontsize=15)
    plt.savefig('../output/plots/crisis_length_hist.png')
    
    #store output csv
    crisis_length_freq_pd.to_csv('crisisLengthOutput.csv')



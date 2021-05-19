import sys
sys.path.insert(0,'../')
import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from Simulation.simulation_benchmark import simulation_model
sys.path.insert(0,'../Models/')
from Benchmark.model_recursive_class import model_recursive
from Benchmark.model_class import model
import os

'''
Variable names used in model input:
rhoE: discount rate of experts
rhoH: discount rate of households
aE: productivity of experts
aH: productivity of households
alpha: skin-in-the-game constraint
kappa: investment costs 
delta: depreciation rate
gammaE: risk aversion of experts
gammaH: risk aversion of households
IES: Inter-temporal elasticity of substitution
zbar: mean proportion of experts 
lambda_d : death rate
utility: type of utility function
nsim: number of simulations to run
'''

if __name__ == '__main__':
    try:
        home = os.path.expanduser('~')
        if not os.path.exists(os.path.join(home,'Desktop','pickles')):
                os.mkdir(os.path.join(home,'Desktop','pickles'))
        pickle_path = os.path.join(home,'Desktop','pickles','') #specify location to store pickles (need lots of space)
    except:
        print('Cannot set path for pickles')
    
    #function to store objects as pickles
    def pickle_stuff(object_name,filename):
        with open(pickle_path+filename,'wb') as f:
            dill.dump(object_name,f)
    #function to read objects 
    def read_pickle(filename):
        with open(str(filename) + '.pkl', 'rb') as f:
            return dill.load(f)
    
    run = 'key' 
    '''
    specify whether to run 'key' or 'all' (or 'base') or 'load and plot'. 'key' runs only for risk aversion 1,2,5, and 15.
    'all' runs for risk aversion from 1 till 20. 'load and plot' loads from pickles and plots equity risk premium.
    '''
    
    if run == 'all':
        params={'rhoE': 0.06, 'rhoH': 0.04, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':10, 'delta':0.02, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':1, 'gammaH':1,'load':False, 'scale':2}
        params['utility'] = 'recursive' #utility can be 'recursive (IES=1)', 'recursive_general(IES!=1)', or 'crra'
        params['nsim']=30
        sims = [0]*22
        for j in range(0,20):
            params['gammaE'] = params['gammaH'] = j+1
            sims[j] = simulation_model(params)
            sims[j].simulate()
            sims[j].compute_statistics()
            print(sims[j].stats)
            sims[j].write_files()
            pickle_stuff(sims[j],  'sim'+ str(j) + '.pkl')
    
    if run == 'key':
        sims = []
        params={'rhoE': 0.06, 'rhoH': 0.04, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':10, 'delta':0.02, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':1, 'gammaH':1,'load':False, 'scale':2}
        params['gammaE'] = 1; params['gammaH'] = 1; params['sigma'] = 0.06; params['utility'] = 'recursive'
        params['nsim']=100
        sim1 = simulation_model(params)
        sim1.simulate()  
        sim1.compute_statistics()
        print(sim1.stats)
        sim1.write_files()
        pickle_stuff(sim1,  str('sim1') + '.pkl')
        
        
        params['gammaE'] = 5; params['gammaH'] = 5;
        sim2 = simulation_model(params)
        sim2.simulate()  
        sim2.compute_statistics()
        print(sim2.stats)
        sim2.write_files()
        pickle_stuff(sim2, str('sim2') + '.pkl')
        
        params['gammaE'] = 10; params['gammaH'] = 10;
        sim3 = simulation_model(params)
        sim3.simulate()  
        sim3.compute_statistics()
        print(sim3.stats)
        sim3.write_files()
        pickle_stuff(sim3, str('sim3') + '.pkl')
        
        
        params['gammaE'] = 20; params['gammaH'] = 20;
        sim4 = simulation_model(params)
        sim4.simulate()  
        sim4.compute_statistics()
        print(sim4.stats)
        sim4.write_files()
        pickle_stuff(sim4, str('sim4') + '.pkl')
        
        for j in range(1,5):
            sims.append('sim'+ str(j))
            
            
    
    #get tables
    vars_ = ['E[theta]', 'E[iota]', 'E[r]', 'E[rp]', 'E[Q]', 'E[sigka]', 'E[mrp]', 'E[mure]', 'Std[iota]', 'Std[rp]','Std[r]', 'Std[mure]',
            'Corr(theta,shock)', 'Corr(retQ,r)', 'Corr(rp,sigka)', 'Corr(mure,r)','prob (dt freq)']
    vars_name = ['E[leverage]','E[inv. rate]', 'E[risk free rate]', 'E[risk premia]', 'E[price]', 'E[return volatility]', 'E[price of risk]','E[expected return]' ,'Std[inv. rate]',
                 'Std[risk premia]', 'Std[risk free rate]','Std[expected return]','Corr(leverage,shock)','Corr(price return, riskf ree rate)','Corr(risk premia, volatility)','Corr(expected return, risk free rate)','prob']
    
    try:
        sims[0]
    except:
        sims = [0]*21
        for i in range(20):
            sims[i] = read_pickle(pickle_path + 'sim' + str(i))
    
    table_merged = pd.concat((sims[0].stats.loc[vars_],
               sims[4].stats.loc[vars_],
               sims[9].stats.loc[vars_],
               sims[19].stats.loc[vars_]),axis=1)
    columns = table_merged.copy().columns
    table_merged = table_merged.T.reset_index(drop=True).T
    format_mapping={'E[theta]': '{:,.2f}', 'E[iota]' : '{:,.2%}', 'E[r]' : '{:,.2%}', 'E[rp]' : '{:,.2%}', 'E[Q]' : '{:,.2f}',\
                    'E[sigka]' : '{:,.2%}', 'E[mrp]' : '{:,.2f}', 'E[mure]' : '{:,.2%}', 'Std[iota]' : '{:,.2%}', 'Std[rp]' : '{:,.2%}', 'Std[r]' : '{:,.2%}', 'Std[mure]' : '{:,.2%}', \
                    'Corr(theta,shock)' : '{:,.2f}', 'Corr(retQ,r)' : '{:,.2f}', 'Corr(rp,sigka)' : '{:,.2f}', 'prob (dt freq)' : '{:,.2%}'}
    for key, value in format_mapping.items():
        table_merged.loc[key] = table_merged.loc[key].apply(value.format)
    table_merged = table_merged.replace(['nan%','nan'], '', regex=True)
    table_merged.index = vars_name
    table_merged.columns = columns
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\begin{document}"""
    endtex = "\end{document}"
    f = open('../output/stats_table.tex', 'w')
    f.write(beginningtex)
    f.write(table_merged.to_latex())
    f.write(endtex)
    f.close()
    
    if run=='load and plot':
        plot_sims = []
        for j in [0,1,2,3,4,19]:
            plot_sims.append('sims['+str(j) +']')
        #del plot_sims[7]
        y,x1,x2, x3,x4 = [], [], [],[],[]
        vars_plot = ['E[rp]','prob (dt freq)','gamma', 'Std[rp]','crisis_length']
        for j in range(len(plot_sims)):
            if j!=6:
                y.append(eval(plot_sims[j] + '.stats.loc[vars_plot[0]].values'))
                x1.append(eval(plot_sims[j] + '.stats.loc[vars_plot[1]].values'))
                x2.append(eval(plot_sims[j] + '.stats.loc[vars_plot[2]].values'))
                x3.append(eval(plot_sims[j] + '.stats.loc[vars_plot[3]].values'))
                x4.append(eval(plot_sims[j] + '.crisis_length'))
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot( np.array(x1)[:,0], np.array(y)[:,0], color = 'green',linestyle='--')
        ax1.scatter(np.array(x1)[0,0], np.array(y)[0,0], label = 'RA = 1', color = 'black', marker = 'x')
        ax1.scatter(np.array(x1)[2,0], np.array(y)[2,0], label = 'RA = 3', color = 'black', marker = 'o')
        ax1.scatter(np.array(x1)[4,0], np.array(y)[4,0], label = 'RA = 5', color = 'black', marker = 'v')
        ax1.scatter(np.array(x1)[-1,0], np.array(y)[-1,0], label = 'RA = 20', color = 'black', marker = 's')
        ax2.plot( np.array(x1)[:,0], np.array(x3)[:,0], label = 'Crisis periods')
        ax2.scatter(np.array(x1)[0,0], np.array(x3)[0,0], color = 'black', marker = 'x')
        ax2.scatter(np.array(x1)[2,0], np.array(x3)[2,0], color = 'black', marker = 'o')
        ax2.scatter(np.array(x1)[4,0], np.array(x3)[4,0], color = 'black', marker = 'v')
        ax2.scatter(np.array(x1)[-1,0], np.array(x3)[-1,0], color = 'black', marker = 's')
        ax1.set_xlabel('Probability of Crisis')
        ax1.set_ylabel('E[risk premia]', color='g')
        ax2.set_ylabel('Std[risk premia]', color='b')
        ax1.legend(bbox_to_anchor=(0.1,1), loc="upper left")
        ax1.grid(False)
        ax2.grid(False)
        plt.savefig('../output/plots/trade_off.png')
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot( np.array(x1)[:,0], np.array(y)[:,1], color = 'green',linestyle='--')
        ax1.scatter(np.array(x1)[0,0], np.array(y)[0,1], label = 'RA = 1', color = 'black', marker = 'x')
        ax1.scatter(np.array(x1)[2,0], np.array(y)[2,1], label = 'RA = 3', color = 'black', marker = 'o')
        ax1.scatter(np.array(x1)[4,0], np.array(y)[4,1], label = 'RA = 5', color = 'black', marker = 'v')
        ax1.scatter(np.array(x1)[-1,0], np.array(y)[-1,1], label = 'RA = 20', color = 'black', marker = 's')
        ax2.plot( np.array(x1)[:,0], np.array(x3)[:,1], label = 'Crisis periods')
        ax2.scatter(np.array(x1)[0,0], np.array(x3)[0,1], color = 'black', marker = 'x')
        ax2.scatter(np.array(x1)[2,0], np.array(x3)[2,1], color = 'black', marker = 'o')
        ax2.scatter(np.array(x1)[4,0], np.array(x3)[4,1], color = 'black', marker = 'v')
        ax2.scatter(np.array(x1)[-1,0], np.array(x3)[-1,1], color = 'black', marker = 's')
        ax1.set_xlabel('Probability of Crisis')
        ax1.set_ylabel('E[rp]', color='g')
        ax2.set_ylabel('Std[rp]', color='b')
        ax1.legend(bbox_to_anchor=(0.1,1), loc="upper left")
        ax1.grid(False)
        ax2.grid(False)
        plt.savefig('../output/plots/trade_off_crisis.png')
        
    ####plot distribution
    
    ####plots for Figure1
    params={'rhoE': 0.06, 'rhoH': 0.04, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':10, 'delta':0.02, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':1, 'gammaH':1,'load':False, 'scale':2}
    
    if run=='key?':
        rec1 = eval(sims[0])
        rec2 = eval(sims[1])
        rec3 = eval(sims[-1])
        
        vars_list = ['Q','Phi','sig_ka','mu_z','sig_za','mrp','rp'] 
        obs = ['log','crra','rec']
        vars_ = []
        for v in vars_list:
            for o in obs:
                vars_.append(str(o) + '.' + str(v))
        labels = ['$q$','$\psi$','$\sigma_q + \sigma$','$\mu^z$','$\sigma^z$','$\zeta_{e}$', '$(\mu_e^R-r)$']
        title = ['Price','Capital Share: Experts', 'Price volatility','Drift of wealth share: Experts',\
                                 'Volatility of wealth share: Experts', 'Market price of risk: Experts','Risk premia: Experts']
        if not os.path.exists('../output/plots'):
            os.mkdir('../output/plots') 
        plot_path = '../output/plots/'            
        for i in range(len(vars_list)):
            last = 20
            plt.plot(rec1.z[1:-last], eval(vars_[3*i])[1:-last,0],label='RA=1')
            plt.plot(rec2.z[1:-last],eval(vars_[3*i+1])[1:-last,0],label='RA=5')
            plt.plot(rec3.z[1:-last],eval(vars_[3*i+2])[1:-last,0],label='RA=15',color='b') 
            plt.grid(True)
            plt.legend(loc=0)
            plt.axis('tight')
            plt.xlabel('Wealth share (z)')
            plt.ylabel(labels[i])
            plt.title(title[i], fontsize = 20)
            plt.rc('legend', fontsize=12) 
            plt.rc('axes',labelsize = 15)
            #plt.rc('figure', titlesize=50)
            plt.savefig(plot_path + str(vars_list[i]) + '_benchmark.png')
            plt.figure()
        
        plt.figure()
        plt.plot(rec1.f_norm[10:-20], label='RA=1')
        plt.plot(rec2.f_norm[10:-20], label='RA=5')
        plt.plot(rec3.f_norm[10:-20], label='RA=15')
        plt.title('Stationary Distribution', fontsize = 20)
    
    
    def plots_distribution(objs):  
        for ob in objs:
            steady_state = np.where(eval(ob).mu_z==min(eval(ob).mu_z, key=lambda x:abs(x-0)))[0][0]
            if True:
                svm = sns.distplot(eval(ob).z_trim_ann.reshape(-1), hist=False, kde=True)
                plt.axvline(eval(ob).z[steady_state], color='k', linestyle='dashed', linewidth=1)
                plt.axvline(eval(ob).z[eval(ob).crisis_z], color='b', linewidth=3)
                fig = svm.get_figure()
                fig.savefig('../output/plots/' + str(ob) + 'distribution_seaborn.png')
                plt.close(fig)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(eval(ob).z_trim_ann.reshape(-1), bins=100)
            ax.axvline(eval(ob).z[steady_state], color='k', linestyle='dashed', linewidth=1)
            ax.axvline(eval(ob).z[eval(ob).crisis_z], color='b', linewidth=3)
            #ax.set_title('Wealth share of experts: stationary distribution', fontsize = 20)
            ax.set_xlabel('Wealth share', fontsize = 15)
            ax.grid('False')
            plt.grid('False')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            plt.gca().axes.get_yaxis().set_visible(False)
            fig.savefig('../output/plots/' + str(ob) + '_distribution.png')
            plt.close(fig)
            
    if run=='all': objs = ['sim1','sim5', 'sim8', 'sim11']
    if run=='key': objs = sims
    if run=='load and plot': objs = plot_sims
    plots_distribution(objs)
        
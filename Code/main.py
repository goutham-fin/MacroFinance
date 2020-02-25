import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from model_class import model
from model_recursive_class import model_recursive



if __name__ == '__main__':
    
    def pickle_stuff(object_name,filename):
        with open(filename,'wb') as f:
            dill.dump(object_name,f)
    def read_pickle(filename):
        with open(str(filename) + '.pkl', 'rb') as f:
            return dill.load(f)
    
    run = 'example' #specify whether to run 'all' or 'example' (or 'base'). 'example' is suggested since 'all' will take several hours to run.
    
    nsim = 500
    
    sims = []
    for j in range(1,21):
        sims.append('sim'+ str(j))
    utility = 'recursive'
    if run == 'example':
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        for j in range(1,len(sims)):
            gammaE = gammaH = j
            sims[j] = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
            sims[j].simulate()
            sims[j].compute_statistics()
            print(sims[j].stats)
            sims[j].write_files()
            pickle_stuff(sims[j], 'sim'+ str(j) + '.pkl')
    
    if run == 'all':
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 1; gammaH = 1; sigma = 0.06; utility = 'recursive'
        sim1 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim1.simulate()  
        sim1.compute_statistics()
        print(sim1.stats)
        sim1.write_files()
        pickle_stuff(sim1, str('sim1') + '.pkl')
        
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 2; gammaH = 2; sigma = 0.06; utility = 'recursive'
        sim2 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim2.simulate()  
        sim2.compute_statistics()
        print(sim2.stats)
        sim2.write_files()
        pickle_stuff(sim2, str('sim2') + '.pkl')

        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 3; gammaH = 3; sigma = 0.06; utility = 'recursive'
        sim3 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim3.simulate()  
        sim3.compute_statistics()
        print(sim3.stats)
        sim3.write_files()
        pickle_stuff(sim3, str('sim3') + '.pkl')
        
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 4; gammaH = 4; sigma = 0.06; utility = 'recursive'
        sim4 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim4.simulate()  
        sim4.compute_statistics()
        print(sim4.stats)
        sim4.write_files()
        pickle_stuff(sim4, str('sim4') + '.pkl')
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 5; gammaH = 5; sigma = 0.06; utility = 'recursive'
        sim5 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim5.simulate()  
        sim5.compute_statistics()
        print(sim5.stats)
        sim5.write_files()
        pickle_stuff(sim5, str('sim5') + '.pkl')
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 7; gammaH = 7; sigma = 0.06; utility = 'recursive'
        sim6 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim6.simulate()  
        sim6.compute_statistics()
        print(sim6.stats)
        sim6.write_files()
        pickle_stuff(sim6, str('sim6') + '.pkl')
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 10; gammaH = 10; sigma = 0.06; utility = 'recursive'
        sim8 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim8.simulate()  
        sim8.compute_statistics()
        print(sim8.stats)
        sim8.write_files()
        pickle_stuff(sim8, str('sim8') + '.pkl')
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 12; gammaH = 12; sigma = 0.06; utility = 'recursive'
        sim9 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim9.simulate()  
        sim9.compute_statistics()
        print(sim9.stats)
        sim9.write_files()
        pickle_stuff(sim9, str('sim9') + '.pkl')
        
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 15; gammaH = 15; sigma = 0.06; utility = 'recursive'
        sim10 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim10.simulate()  
        sim10.compute_statistics()
        print(sim10.stats)
        sim10.write_files()
        pickle_stuff(sim10, str('sim10') + '.pkl')
        
      
        rhoE = 0.06; rhoH = 0.03; aE = 0.10; aH = 0.03;  alpha = 0.5;  kappa = 10; zbar = 0.1; lambda_d = 0.015; sigma = 0.06
        delta = 0.025; #to match E[GDP Growth] = 2.0%
        gammaE = 20; gammaH = 20; sigma = 0.06; utility = 'recursive'
        sim11 = simulation_model(rhoE, rhoH, aE, aH, sigma, alpha, gammaE, gammaH, kappa, delta, lambda_d, zbar, utility, nsim)
        sim11.simulate()  
        sim11.compute_statistics()
        print(sim11.stats)
        sim11.write_files()
        pickle_stuff(sim11, str('sim11') + '.pkl')
        
    
    #get tables
    vars_ = ['E[theta]', 'E[iota]', 'E[r]', 'E[rp]', 'E[Q]', 'E[sigka]', 'E[mrp]', 'E[mure]', 'Std[iota]', 'Std[rp]','Std[r]', 'Std[mure]',
            'Corr(theta,shock)', 'Corr(retQ,r)', 'Corr(rp,sigka)', 'Corr(mure,r)','prob (dt freq)']
    
    
    table_merged = pd.concat((sim1.stats.loc[vars_],
               sim5.stats.loc[vars_],
               sim8.stats.loc[vars_],
               sim11.stats.loc[vars_]),axis=1)
    columns = table_merged.copy().columns
    table_merged = table_merged.T.reset_index(drop=True).T
    format_mapping={'E[theta]': '{:,.2f}', 'E[iota]' : '{:,.2%}', 'E[r]' : '{:,.2%}', 'E[rp]' : '{:,.2%}', 'E[Q]' : '{:,.2f}',\
                    'E[sigka]' : '{:,.2%}', 'E[mrp]' : '{:,.2f}', 'E[mure]' : '{:,.2%}', 'Std[iota]' : '{:,.2%}', 'Std[rp]' : '{:,.2%}', 'Std[r]' : '{:,.2%}', 'Std[mure]' : '{:,.2%}', \
                    'Corr(theta,shock)' : '{:,.2f}', 'Corr(retQ,r)' : '{:,.2f}', 'Corr(rp,sigka)' : '{:,.2f}', 'prob (dt freq)' : '{:,.2%}'}
    for key, value in format_mapping.items():
        table_merged.loc[key] = table_merged.loc[key].apply(value.format)
    table_merged = table_merged.replace(['nan%','nan'], '', regex=True)
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
    
    
    plot_sim = []

    for j in range(1,12):
        plot_sim.append('sim'+ str(j))
    del plot_sim[7]
    y,x1,x2, x3,x4 = [], [], [],[],[]
    vars_plot = ['E[rp]','prob (dt freq)','gamma', 'Std[rp]','crisis_length']
    for j in range(len(plot_sim)):
        if j!=6:
            y.append(eval(plot_sim[j] + '.stats.loc[vars_plot[0]].values'))
            x1.append(eval(plot_sim[j] + '.stats.loc[vars_plot[1]].values'))
            x2.append(eval(plot_sim[j] + '.stats.loc[vars_plot[2]].values'))
            x3.append(eval(plot_sim[j] + '.stats.loc[vars_plot[3]].values'))
            x4.append(eval(plot_sim[j] + '.crisis_length'))
    x4 = [0 if math.isnan(i) else i for i in x4]        
    #crisis length
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x4)
    ax2.plot(np.array(y)[:,0])       
    plt.xlabel('Risk Aversion', fontsize = 15)
    ax1.set_ylabel('Average crisis length (in months)', fontsize = 15)
    ax2.set_ylabel('Expected Risk Premia', fontsize = 15)
    plt.savefig('../output/plots/crisis_length.png')
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot( np.array(x1)[:,0], np.array(y)[:,0], label = 'All periods')
    plt.scatter(np.array(x1)[0,0], np.array(y)[0,0], label = 'RA = 1', color = 'black', marker = 'x')
    plt.scatter(np.array(x1)[2,0], np.array(y)[2,0], label = 'RA = 3', color = 'black', marker = 'o')
    plt.scatter(np.array(x1)[4,0], np.array(y)[4,0], label = 'RA = 5', color = 'black', marker = 'v')
    plt.scatter(np.array(x1)[-1,0], np.array(y)[-1,0], label = 'RA = 20', color = 'black', marker = 's')
    plt.plot( np.array(x1)[:,0], np.array(x3)[:,0], label = 'Crisis periods')
    plt.scatter(np.array(x1)[0,0], np.array(x3)[0,0], color = 'black', marker = 'x')
    plt.scatter(np.array(x1)[2,0], np.array(x3)[2,0], color = 'black', marker = 'o')
    plt.scatter(np.array(x1)[4,0], np.array(x3)[4,0], color = 'black', marker = 'v')
    plt.scatter(np.array(x1)[-1,0], np.array(x3)[-1,0], color = 'black', marker = 's')
    #plt.ylim(0.0,0.22)
    plt.legend(loc=1)
    plt.ylabel('Expected Risk Premia')
    plt.xlabel('Probability of Crisis')
    plt.rc('legend', fontsize=12) 
    plt.rc('axes',labelsize = 15)
    plt.savefig('../output/plots/trade_off_er.png')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot( np.array(x1)[:,0], np.array(x3)[:,0], label = 'All periods')
    plt.scatter(np.array(x1)[0,0], np.array(x3)[0,0], label = 'RA = 1', color = 'black', marker = 'x')
    plt.scatter(np.array(x1)[2,0], np.array(x3)[2,0], label = 'RA = 3', color = 'black', marker = 'o')
    plt.scatter(np.array(x1)[4,0], np.array(x3)[4,0], label = 'RA = 5', color = 'black', marker = 'v')
    plt.scatter(np.array(x1)[-1,0], np.array(x3)[-1,0], label = 'RA = 20', color = 'black', marker = 's')
    plt.plot( np.array(x1)[:-4,0], np.array(x3)[:-4,1], label = 'Crisis periods')
    plt.scatter(np.array(x1)[0,0], np.array(x3)[0,1], color = 'black', marker = 'x')
    plt.scatter(np.array(x1)[2,0], np.array(x3)[2,1], color = 'black', marker = 'o')
    plt.scatter(np.array(x1)[4,0], np.array(x3)[4,1], color = 'black', marker = 'v')
    plt.scatter(np.array(x1)[-1,0], np.array(x3)[-1,1], color = 'black', marker = 's')
    #plt.ylim(0.0,0.22)
    plt.legend(loc=1)
    plt.ylabel('Std Risk Premia')
    plt.xlabel('Probability of Crisis')
    plt.rc('legend', fontsize=12) 
    plt.rc('axes',labelsize = 15)
    plt.savefig('../output/plots/trade_off_sd.png')
   
    
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
    plt.savefig('../output/plots/trade_off_crisis.png')
    
    ####plot distribution
    
    def plots_distribution(objs):  
        for ob in objs:
            svm = sns.distplot(eval(ob).z_trim_ann.reshape(-1), hist=False, kde=True)
            plt.axvline(eval(ob).z_trim_ann.reshape(-1).mean(), color='k', linestyle='dashed', linewidth=1)
            plt.axvline(eval(ob).z[eval(ob).crisis_z], color='b', linewidth=3)
            fig = svm.get_figure()
            fig.savefig('../output/plots/' + str(ob) + 'distribution_seaborn.png')
            plt.close(fig)
            fig = plt.figure()
            
            ax = fig.add_subplot(111)
            ax.hist(eval(ob).z_trim_ann.reshape(-1), bins=100)
            ax.axvline(eval(ob).z_trim_ann.reshape(-1).mean(), color='k', linestyle='dashed', linewidth=1)
            ax.axvline(eval(ob).z[eval(ob).crisis_z], color='b', linewidth=3)
            #ax.set_title('Wealth share of experts: stationary distribution', fontsize = 20)
            ax.set_xlabel('Wealth share', fontsize = 15)
            fig.tight_layout()
            plt.gca().axes.get_yaxis().set_visible(False)
            fig.savefig('../output/plots/' + str(ob) + '_distribution.png')
            plt.close(fig)
            
    objs = ['sim1','sim5', 'sim8', 'sim11']
    plots_distribution(objs)
        
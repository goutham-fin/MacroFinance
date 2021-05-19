import sys
sys.path.insert(0, '../Models/')
from Benchmark.model_class import model
from Benchmark.model_recursive_class import model_recursive
from Benchmark.model_recursive_general_class import model_recursive_general
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import dill
import statsmodels.tsa.api as tsa
import statsmodels.regression.linear_model as sm
import statsmodels.api as sm_filters
from statsmodels.api import add_constant
from statsmodels.iolib.summary2 import summary_col
import statistics
from itertools import groupby
from statsmodels.graphics.tsaplots import plot_acf


class simulation_model():
    '''
    This class simulates the models from generalized framework built in the paper 
    Confronting Macro-finance model with data (2020)
    For log utility: set utility='recursive'; IES=1; gammaE=1; gammaH=1
    For CRRA utility: set utility ='crra'; 
    For Recursive utility with IES=1: set utility='recursive';IES=1
    For Recursive utilty with non-unitary IES: set utility = 'recursive_general'; 
    '''
    def __init__(self, params):
        self.params = params
        T = 5000
        self.dt = 1/12
        self.t = np.arange(0,T,self.dt)
        self.burn_period = 1000/self.dt
        
        self.mu0 = 0.5
        self.k0 = 100
        
        def read_pickle(filename):
            with open(str(filename) + '.pkl', 'rb') as f:
                return dill.load(f)
        if self.params['load']==True:
            self.bk = read_pickle(self.params['pickle_name'])
        else:
            if self.params['utility'] == 'recursive' or self.params['utility'] == 'recursive_general':
                if self.params['utility'] == 'recursive':
                    self.bk = model_recursive(self.params)
                    
                if self.params['utility'] == 'recursive_general':
                    self.bk = model_recursive_general(self.params)
    
                
            elif self.params['utility'] == 'crra':
                self.bk = model(self.params)
                self.bk.solve()
                
            self.bk.solve()
        self.z, self.crisis_z,self.mu_z, self.sig_za, self.sig_ka, self.iota, self.theta, self.thetah, self.rp, self.rp_, self.r, self.Q, self.rho, self.rho_, self.Phi, self.lambda_k, self.s_a = self.bk.z, self.bk.thresholdIndex, self.bk.mu_z, self.bk.sig_za, self.bk.ssq, self.bk.iota, \
                                                               self.bk.theta, self.bk.thetah, self.bk.rp, self.bk.rp_, self.bk.r, self.bk.q, self.bk.params['rhoE'], self.bk.params['rhoH'], self.bk.Phi, self.bk.params['delta'], self.bk.params['sigma']
        self.mrp = self.bk.priceOfRiskE
        self.mrph = self.bk.priceOfRiskH
        self.A, self.AminusIota, self.pd = self.bk.A, self.bk.AminusIota, self.bk.pd
        self.f = self.bk.f
        self.mu_rh = self.bk.mu_rH
        self.mu_re = self.bk.mu_rE
        self.psi = self.bk.psi
        self.f_norm = self.bk.f_norm
        self.consWealthRatioE = self.bk.consWealthRatioE
        self.consWealthRatioH = self.bk.consWealthRatioH
            
        if not os.path.exists('../output'):
            os.mkdir('../output')  
                                                    
    def interpolate_values(self):
        #interpolate variables outside the grid points
        self.mu_z_fn = self.interpolate_var(self.mu_z[:,0])
        self.sig_z_fn = self.interpolate_var(self.sig_za[:,0])
        self.iota_fn = self.interpolate_var(self.iota[:,0])
        self.r_fn = self.interpolate_var(self.r[:,0])
        self.thetah_fn = self.interpolate_var(self.thetah[:,0])
        self.theta_fn = self.interpolate_var(self.theta[:,0])
        self.rp_fn = self.interpolate_var(self.rp[:,0])
        self.rph_fn = self.interpolate_var(self.rp_[:,0])
        self.sig_ka_fn = self.interpolate_var(self.sig_ka[:,0])
        self.Q_fn =  self.interpolate_var(self.Q[:,0])
        self.Phi_fn = self.interpolate_var(self.Phi[:,0])
        self.mrp_fn = self.interpolate_var(self.mrp[:,0])
        self.mrph_fn = self.interpolate_var(self.mrph[:,0])
        self.AminusIota_fn = self.interpolate_var(self.AminusIota[:,0])
        self.A_fn = self.interpolate_var(self.A[:,0])
        self.pd_fn = self.interpolate_var(self.pd[:,0])
        self.mure_fn = self.interpolate_var(self.mu_re[:,0])

    def interpolate_var(self,var):
        return interp1d(self.z,var, kind='cubic')
         
    def simulate(self):
        self.interpolate_values()
        #self.params['nsim'] = 3
        self.z_sim = np.array(np.tile(self.mu0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.k_sim = np.array(np.tile(self.k0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        self.shock_series = np.array(np.tile(0,(self.t.shape[0],self.params['nsim'])),dtype=np.float64)
        
        for n in range(self.params['nsim']):    
            for i in range(1,self.t.shape[0]):
                shock = np.random.normal(0,1)
                self.shock_series[i,n] = shock
                self.z_sim[i,n] = self.z_sim[i-1,n] + self.mu_z_fn(self.z_sim[i-1,n])*self.dt + self.sig_z_fn(self.z_sim[i-1,n]) *shock *np.sqrt(self.dt)
                self.k_sim[i,n] = self.k_sim[i-1,n] + self.k_sim[i-1,n]*(self.Phi_fn(self.z_sim[i-1,n]) - self.lambda_k) * self.dt + self.k_sim[i-1,n] * self.s_a * shock * np.sqrt(self.dt)
                if self.z_sim[i,n] < 0.001: #reflecting boundary at epsilon, instead of at zero
                    self.z_sim[i,n] = 2*0.001 - self.z_sim[i,n]
                elif self.z_sim[i,n] > 0.999: #reflecting boundary at 1-epsilon, instead of at 1
                    self.z_sim[i,n] = 2*0.999- self.z_sim[i,n]
                
            if n%100==0:
                print(n)
        '''
        The variables xxx_trim stores simulated array after removing the burnout period
        The variables xxx_trim_ann stores annualized values
        '''
        self.z_trim = self.z_sim[int(self.burn_period):,:]
        self.k_trim = self.k_sim[int(self.burn_period):,:]
        self.shock_trim = self.shock_series[int(self.burn_period):,:]
        self.z_trim_ann = np.full([ int(self.z_trim.shape[0]*self.dt),int(self.z_trim.shape[1])],np.nan)
        self.k_trim_ann = np.full([ int(self.k_trim.shape[0]*self.dt),int(self.k_trim.shape[1])],np.nan)
        self.shock_trim_ann = np.full([ int(self.z_trim.shape[0]*self.dt),int(self.z_trim.shape[1])],np.nan)
        for j in range(self.z_trim_ann.shape[1]):
            self.z_trim_ann[:,j] = self.z_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.shock_trim_ann[:,j] = self.shock_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)
            self.k_trim_ann[:,j] = self.k_trim[:,j].reshape(-1,int(1/self.dt)).sum(axis=1)  #sum capital 
        self.z_sim_avg = self.z_trim_ann.mean(axis=0)
        '''
        if prob. of  staying in crisis region (annual frequency) is needed use the below line instead
        self.prob = ((self.z_trim_ann.reshape(-1) < self.z[self.crisis_z])/self.z_trim_ann.reshape(-1).shape[0]).sum()
        '''
        self.prob = ((self.z_trim.reshape(-1) < self.z[self.crisis_z])/self.z_trim.reshape(-1).shape[0]).sum()
        self.crisis_count = [0]
        for j in range(self.params['nsim']): #annualized z_trim
            self.crisis_count.append(np.sum(self.z_trim[:,j].reshape(-1,int(1/self.dt)).mean(axis=1)<self.z[self.crisis_z]))
        self.crisis_indicator = np.where(self.z_trim_ann < self.z[self.crisis_z],1,0.0)
        self.crisis_indicator[self.crisis_indicator==0]=np.nan
        self.crisis_indicator_freq = np.where(self.z_trim < self.z[self.crisis_z],1,0.0)
        self.crisis_indicator_freq[self.crisis_indicator_freq==0] = np.nan
        self.crisis_length =  np.array([(k, sum(1 for i in g)) for k,g in groupby(self.crisis_indicator.transpose().reshape(-1))])
        self.crisis_length_mean = np.mean(self.crisis_length[~np.isnan(self.crisis_length[:,0])][:,1])
        self.crisis_length_freq =  np.array([(k, sum(1 for i in g)) for k,g in groupby(self.crisis_indicator_freq.transpose().reshape(-1))])
        self.crisis_length_freq_mean = np.mean(self.crisis_length_freq[~np.isnan(self.crisis_length_freq[:,0])][:,1])
        self.crisis_length_freq_data = pd.DataFrame(self.crisis_length_freq[~np.isnan(self.crisis_length_freq[:,0])][:,1])
        
        
        try:
            plt.hist(self.z_trim_ann.reshape(-1), bins=100)
        except:
            print('0 crisis points')
        plt.figure()

        self.crisis_z_freq = self.z_trim_ann*self.crisis_indicator
        self.crisis_z_freq = self.crisis_z_freq.reshape(-1)
        
    def compute_statistics_ann_fn(self,fn):
        temp_all = fn(self.z_trim_ann.transpose().reshape(-1)).reshape(self.params['nsim'],-1).transpose()
        temp_crisis = (fn(self.z_trim_ann.transpose().reshape(-1))*self.crisis_indicator.transpose().reshape(-1)).reshape(self.params['nsim'],-1).transpose()
        temp_good = (fn(self.z_trim_ann.transpose().reshape(-1)) * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan)).transpose().reshape(-1)).reshape(self.params['nsim'],-1).transpose()
        return temp_all , temp_crisis , temp_good
    
    def compute_statistics_fn(self,fn):
        temp_all = fn(self.z_trim.transpose().reshape(-1)).reshape(self.params['nsim'],-1).transpose()
        temp_crisis= (fn(self.z_trim)*self.crisis_indicator_freq)
        temp_good = (fn(self.z_trim) * np.array(pd.DataFrame(1-self.crisis_indicator_freq).fillna(1).replace(0,np.nan)))
        return temp_all, temp_crisis, temp_good 
            
    def compute_statistics_ann_var(self,temp_all):
            temp_crisis = temp_all * self.crisis_indicator
            temp_good = temp_all * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan))
            return temp_crisis, temp_good
         
    def compute_statistics(self):
        self.theta_fn_all, self.theta_fn_crisis, self.theta_fn_good = self.compute_statistics_ann_fn(self.theta_fn)
        self.thetah_fn_all, self.thetah_fn_crisis, self.thetah_fn_good = self.compute_statistics_ann_fn(self.thetah_fn)
        self.iota_fn_all, self.iota_fn_crisis, self.iota_fn_good = self.compute_statistics_ann_fn(self.iota_fn)
        self.r_fn_all, self.r_fn_crisis, self.r_fn_good = self.compute_statistics_ann_fn(self.r_fn)
        self.rp_fn_all, self.rp_fn_crisis, self.rp_fn_good = self.compute_statistics_ann_fn(self.rp_fn)
        self.rph_fn_all, self.rph_fn_crisis, self.rph_fn_good = self.compute_statistics_ann_fn(self.rph_fn)
        self.Q_fn_all, self.Q_fn_crisis, self.Q_fn_good = self.compute_statistics_ann_fn(self.Q_fn)
        self.sigka_fn_all, self.sigka_fn_crisis, self.sigka_fn_good = self.compute_statistics_ann_fn(self.sig_ka_fn)
        self.mrp_fn_all, self.mrp_fn_crisis, self.mrp_fn_good = self.compute_statistics_ann_fn(self.mrp_fn)
        self.mrph_fn_all, self.mrph_fn_crisis, self.mrph_fn_good = self.compute_statistics_ann_fn(self.mrph_fn)
        self.Phi_fn_all, self.Phi_fn_crisis, self.Phi_fn_good = self.compute_statistics_ann_fn(self.Phi_fn)
        self.AminusIota_fn_all, self.AminusIota_fn_crisis, self.AminusIota_fn_good = self.compute_statistics_ann_fn(self.AminusIota_fn)
        self.A_fn_all, self.A_fn_crisis, self.A_fn_good = self.compute_statistics_ann_fn(self.A_fn)
        self.pd_fn_all, self.pd_fn_crisis, self.pd_fn_good = self.compute_statistics_ann_fn(self.pd_fn)
        self.mure_fn_all, self.mure_fn_crisis, self.mure_fn_good = self.compute_statistics_ann_fn(self.mure_fn)
        
        self.shock_fn_all = self.shock_trim_ann
        self.shock_fn_crisis, self.shock_fn_good = self.compute_statistics_ann_var(self.shock_fn_all)
        self.retQ_fn_all = np.diff(np.log(self.Q_fn_all),axis=0)
        self.retQ_fn_all = np.vstack([self.retQ_fn_all[0,:],self.retQ_fn_all])
        self.retQ_fn_crisis, self.retQ_fn_good = self.compute_statistics_ann_var(self.retQ_fn_all)
        self.output_fn_all = self.A_fn_all * self.k_trim_ann
        self.outputG_fn_all = np.array(np.tile(0,(self.output_fn_all.shape[0],self.params['nsim'])), dtype=np.float64);
        for n in range(self.params['nsim']):
            self.outputG_fn_all[:,n] = np.hstack([np.nan,np.diff(np.log(self.output_fn_all[:,n]))])
        self.outputG_fn_crisis, self.outputG_fn_good = self.outputG_fn_all*self.crisis_indicator, self.outputG_fn_all * np.array(pd.DataFrame(1-self.crisis_indicator).fillna(1).replace(0,np.nan))
        
        
        
        vars_corr = ['theta_shock','theta_Q','theta_mrp','shock_mrp','Q_r', 'retQ_r', 'rp_sigka', 'shock_sigka', 'mure_r'] #rp_outputG
        self.stats_corr = pd.DataFrame(np.full([len(vars_corr),3],np.nan))
        #correlation statistics
        for i in range(len(vars_corr)):
            temp_corr_all, temp_corr_crisis, temp_corr_good = [],[],[]
            for sim in range(self.params['nsim']):
                temp_corr_all.append(np.corrcoef(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_all').transpose()[sim],\
                                                 eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_all').transpose()[sim])[1][0])
                try:
                    temp_corr_crisis.append(np.corrcoef(np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_crisis')[:,sim]).dropna()).transpose(),\
                                                        np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_crisis')[:,sim]).dropna()).transpose())[1][0])
                except:
                    pass
                try:
                    temp_corr_good.append(np.corrcoef(np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[0]) + '_fn_good')[:,sim]).dropna()).transpose(),\
                                                        np.array(pd.DataFrame(eval('self.'+ str(vars_corr[i].split('_')[1]) + '_fn_good')[:,sim]).dropna()).transpose())[1][0])
                except:
                    pass
            self.stats_corr.iloc[i,0] = np.nanmean(temp_corr_all)
            self.stats_corr.iloc[i,1] = np.nanmean(temp_corr_crisis)
            self.stats_corr.iloc[i,2] = np.nanmean(temp_corr_good)
        
        index_temp = []
        for j in range(len(vars_corr)): index_temp.append('Corr(' + str(vars_corr[j].split('_')[0]) + ',' + str(vars_corr[j].split('_')[1]) + ')')
        self.stats_corr.index = index_temp  
        del index_temp
        
        
        #compute relevant statistics
        vars = ['theta','thetah', 'iota', 'r', 'rp', 'rph', 'Q', 'sigka', 'mrp', 'mrph', 'pd', 'mure','outputG', 'retQ']
        self.stats = pd.DataFrame(np.zeros([len(vars)*2,3]))
        for i in range(len(vars)):
            self.stats.iloc[i,0] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_all')),3)
            self.stats.iloc[i,1] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_crisis')),3)
            self.stats.iloc[i,2] = round(np.nanmean(eval('self.' + str(vars[i]) + '_fn_good')),3)
        for i in range(len(vars)):
            std_temp_all, std_temp_crisis, std_temp_good = [], [], []
            for sim in range(self.params['nsim']):
                std_temp_all.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_all').transpose()[sim]))
                std_temp_crisis.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_crisis').transpose()[sim]))
                std_temp_good.append(np.nanstd(eval('self.' + str(vars[i]) + '_fn_good').transpose()[sim]))
            
            self.stats.iloc[i+len(vars),0] = np.nanmean(std_temp_all)
            self.stats.iloc[i+len(vars),1] = np.nanmean(std_temp_crisis)
            self.stats.iloc[i+len(vars),2] = np.nanmean(std_temp_good)
        index_temp = []
        for j in range(len(vars)): index_temp.append('E[' + str(vars[j]) + ']')        
        for j in range(len(vars)): index_temp.append('Std[' + str(vars[j]) + ']')
        self.stats.index = index_temp
        del index_temp
        self.stats = pd.concat((self.stats,self.stats_corr))
        
        self.stats =  pd.concat((self.stats, pd.Series({'No of crisis periods': self.Q_fn_crisis.shape[0], 'freq': int(1/self.dt), 'prob (dt freq)': self.prob,  'gamma':self.params['gammaE'],'sigma':self.params['sigma'], 
                                                        'a':self.params['aE'], 'ah':self.params['aH'], 'depr':self.params['delta'], 'kappa':self.params['kappa'],'alpha':self.params['alpha']}).transpose()), axis=0, ignore_index = False)
        self.stats.columns = ['All','Crisis','Good']
        self.stats = round(self.stats,4)
        
        #time series momentum (optional, not included in the paper)
        self.Q_freq_fn_all, self.Q_freq_fn_crisis, self.Q_freq_fn_good = self.compute_statistics_fn(self.Q_fn)
        self.Phi_freq_fn_all, self.Phi_freq_fn_crisis, self.Phi_freq_fn_good = self.compute_statistics_fn(self.Phi_fn)
        self.sigka_freq_fn_all, self.sigka_freq_fn_crisis, self.sigka_freq_fn_good = self.compute_statistics_fn(self.sig_ka_fn)
        self.theta_freq_fn_all, self.theta_freq_fn_crisis, self.theta_freq_fn_good = self.compute_statistics_fn(self.theta_fn)
        self.iota_freq_fn_all, self.iota_freq_fn_crisis, self.iota_freq_fn_good = self.compute_statistics_fn(self.iota_fn)
        self.r_freq_fn_all, self.r_freq_fn_crisis, self.r_freq_fn_good = self.compute_statistics_fn(self.r_fn)
        self.AminusIota_freq_fn_all, self.AminusIota_freq_fn_crisis, self.AminusIota_freq_fn_good = self.compute_statistics_fn(self.AminusIota_fn)
        self.dexret_freq_fn_all = (np.vstack([np.full([1,self.params['nsim']],np.nan),np.diff(self.Q_freq_fn_all*self.k_trim,axis=0)]) + self.AminusIota_freq_fn_all*self.dt*self.k_trim)/(self.Q_freq_fn_all*self.k_trim) - self.r_freq_fn_all*self.dt
        self.iota_freq_fn_all,_,_ = self.compute_statistics_fn(self.iota_fn)
        self.exret_freq_fn_all = np.array(pd.DataFrame(self.dexret_freq_fn_all).rolling(2).sum())
        self.rp_freq_fn_all, self.rp_freq_fn_crisis, self.rp_freq_fn_good = self.compute_statistics_fn(self.rp_fn)
        
        self.output_freq_fn_all = self.AminusIota_freq_fn_all * self.k_trim
        self.outputG_freq_fn_all = np.array(np.tile(0,(self.output_freq_fn_all.shape[0],self.params['nsim'])), dtype=np.float64);
        for n in range(self.params['nsim']):
            self.outputG_freq_fn_all[:,n] = np.hstack([np.nan,np.diff(np.log(self.output_freq_fn_all[:,n]))])
        self.outputG_freq_fn_crisis, self.outputG_freq_fn_good = self.outputG_freq_fn_all*self.crisis_indicator_freq, self.outputG_freq_fn_all * np.array(pd.DataFrame(1-self.crisis_indicator_freq).fillna(1).replace(0,np.nan))
        
        self.coeff1, self.coeff2 = [], []
        self.tstat1, self.tstat2 = [], []
        self.coeff_unc, self.tstat_unc = [],[]
        lags = np.arange(1,50)
        for i in range(len(lags)):
            coeff1_temp, coeff2_temp = [],[]
            tstat1_temp, tstat2_temp = [], []
            tstat_unc_temp, coeff_unc_temp = [],[]
            
            for j in range(self.params['nsim']):
                indep_unc = self.exret_freq_fn_all[:,j]
                indep = np.column_stack([self.exret_freq_fn_all[:,j], (self.exret_freq_fn_all[:,j] * np.array(pd.DataFrame(self.crisis_indicator_freq[:,j]).fillna(0)).transpose()).transpose()])
                exog = sm_filters.add_constant(indep)
                exog_unc = sm_filters.add_constant(indep_unc)
                try:
                    reg = sm.OLS(self.exret_freq_fn_all[lags[i]:,j],exog[0:-lags[i],:],missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags' : int(0.75* exog.shape[0]**0.33)}) #int(0.75* exog.shape[0]**0.33)
                    coeff1_temp.append(reg.params[1])
                    coeff2_temp.append(reg.params[2])
                    tstat1_temp.append(reg.params[1]/reg.bse[1])
                    tstat2_temp.append(reg.params[2]/reg.bse[2])
                except:
                    pass
                reg_unc = sm.OLS(self.exret_freq_fn_all[lags[i]:,j],exog_unc[0:-lags[i],:],missing='drop').fit(cov_kwds={'maxlags' : int(0.75* exog.shape[0]**0.1)})
                coeff_unc_temp.append(reg_unc.params[1])
                tstat_unc_temp.append(reg_unc.params[1]/reg_unc.bse[1])
                
                
            self.coeff1.append(np.mean(coeff1_temp))
            self.coeff2.append(np.mean(coeff2_temp))
            self.tstat1.append(np.mean(tstat1_temp))
            self.tstat2.append(np.mean(tstat2_temp))
            self.coeff_unc.append(np.mean(coeff_unc_temp))
            self.tstat_unc.append(np.mean(tstat_unc_temp))
        
  
        
    def write_files(self):
        self.stats.to_csv('../output/stats_bench_gamma' + str(self.params['gammaE']) + '_sigma' + str(self.params['sigma']) + '_' + str(self.params['utility'])  +  '.csv')
        '''
        save figures in the output directory
        '''
        self.plots_()
     
    def plots_(self):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(self.z_trim_ann.reshape(-1), bins=100)
        ax.axvline(self.z_trim_ann.reshape(-1).mean(), color='k', linestyle='dashed', linewidth=1)
        ax.axvline(self.z[self.crisis_z], color='b', linewidth=3)
        ax.set_title('Wealth share of experts: stationary distribution')
        ax.set_xlabel('Wealth share')
        fig.tight_layout()
        fig.savefig('../output/plots/zsim_bench_gamma' + str(self.params['gammaE']) + '_gammah' + str(self.params['gammaH']) + '_sigma' + str(self.params['sigma']) + '_' + str(self.params['utility']) + '.png')
        plt.close(fig)
        
        
        svm = sns.distplot(self.z_trim_ann.reshape(-1), hist=False, kde=True)
        plt.axvline(self.z_trim_ann.reshape(-1).mean(), color='k', linestyle='dashed', linewidth=1)
        plt.axvline(self.z[self.crisis_z], color='b', linewidth=3)
        fig = svm.get_figure()
        fig.savefig('../output/plots/zsim_bench_kde_gamma' + str(self.params['gammaE']) + '_gammah' + str(self.params['gammaH']) + '_sigma' + str(self.params['sigma']) + '_' + str(self.params['utility']) + '.png')
        plt.close(fig)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.z_trim[:,0].reshape(-1,int(1/self.dt)))
        ax.set_title('Sample path of wealth share')
        ax.set_xlabel('Time (annual)')
        ax.set_ylabel('Wealth share of experts')
        fig.savefig('../output/plots/zsim_bench_sample_path_gamma' + str(self.params['gammaE']) + '_gammah' + str(self.params['gammaH']) + '_sigma' + str(self.params['sigma']) + '_' + str(self.params['utility']) + '.png')
        plt.close(fig)
        
        
        #plot with kfe
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.hist(self.z_trim_ann.reshape(-1),bins=1000)
        ax2.plot(self.z[10:-20], self.f[10:-20])
        fig.savefig('../output/plots/zsim_bench_gamma' + str(self.params['gammaE']) + '_gammah' + str(self.params['gammaH']) + '_sigma' + str(self.params['sigma']) + '_' + str(self.params['utility']) + '_kfe' + '.png')
        plt.close(fig)

###########################################################################################
###########################################################################################
###########################################################################################
if __name__ == '__main__':
    params={'rhoE': 0.06, 'rhoH': 0.04, 'aE': 0.11, 'aH': 0.03,
            'alpha':0.5, 'kappa':10, 'delta':0.02, 'zbar':0.1, 
            'lambda_d':0.03, 'sigma':0.06, 'gammaE':1, 'gammaH':1}
    params['IES']=1.0; params['utility'] = 'recursive'; params['nsim']=1000
    params['load']=False
    params['scale']=2
    params['pickle_name']='model1D'
    sim1 = simulation_model(params)
    sim1.simulate()
    sim1.compute_statistics()
    #sim1.write_files()
    print(sim1.stats)
    
    print(sim1.crisis_length_freq_mean)
    print(sim1.crisis_length_freq_data.quantile(0.1))
    print(sim1.crisis_length_freq_data.quantile(0.5))
    print(sim1.crisis_length_freq_data.quantile(0.9))
    
    plt.hist(sim1.crisis_z_freq,bins=100,density=True)
    plt.grid(False)
    #plt.title('Crisis region')
    plt.yticks([])
    plt.xlabel('Wealth share',fontsize=15)
    plt.savefig('../output/plots/crisis_z.png')
    
    
    
    
    
    
        
    
    
    
    
        
        